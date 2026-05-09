import React, { useRef, useState, useEffect } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, Animated,
  Dimensions, ActivityIndicator, Platform,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';

import { COLORS, SPACING, RADIUS, FONT } from '../theme';
import { predictLeaf } from '../services/api';
import type { RootStackParamList } from '../navigation';

const { width, height } = Dimensions.get('window');
const FRAME_SIZE = Math.min(width * 0.75, 300);
const CORNER = 28;

type NavProp = NativeStackNavigationProp<RootStackParamList>;

export default function ScanScreen() {
  const navigation = useNavigation<NavProp>();
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing]     = useState<'back' | 'front'>('back');
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError]         = useState<string | null>(null);

  const cameraRef  = useRef<CameraView>(null);
  const scanLineY  = useRef(new Animated.Value(0)).current;
  const captureScale = useRef(new Animated.Value(1)).current;
  const overlayOpacity = useRef(new Animated.Value(0)).current;

  // Animación de línea de escaneo
  useEffect(() => {
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(scanLineY, { toValue: 1, duration: 2200, useNativeDriver: true }),
        Animated.timing(scanLineY, { toValue: 0, duration: 2200, useNativeDriver: true }),
      ]),
    );
    anim.start();
    return () => anim.stop();
  }, []);

  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={COLORS.gold} />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionIcon}>📷</Text>
        <Text style={styles.permissionTitle}>Acceso a la Cámara</Text>
        <Text style={styles.permissionBody}>
          DeepVine necesita la cámara para fotografiar{'\n'}la hoja de vid y analizarla con IA.
        </Text>
        <TouchableOpacity style={styles.permissionBtn} onPress={requestPermission}>
          <Text style={styles.permissionBtnText}>Conceder permiso</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Captura y análisis ──────────────────────────────────────────────────

  async function handleCapture() {
    if (!cameraRef.current || analyzing) return;
    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    } catch (_) {}

    // Animación del botón
    Animated.sequence([
      Animated.timing(captureScale, { toValue: 0.88, duration: 100, useNativeDriver: true }),
      Animated.timing(captureScale, { toValue: 1.0,  duration: 100, useNativeDriver: true }),
    ]).start();

    try {
      setError(null);
      setAnalyzing(true);
      Animated.timing(overlayOpacity, { toValue: 1, duration: 300, useNativeDriver: true }).start();

      const photo = await cameraRef.current.takePictureAsync({ quality: 0.85, exif: false });
      if (!photo?.uri) throw new Error('No se pudo capturar la imagen.');

      await navigateToResult(photo.uri);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Error desconocido';
      setError(msg);
      setAnalyzing(false);
      Animated.timing(overlayOpacity, { toValue: 0, duration: 300, useNativeDriver: true }).start();
    }
  }

  async function handleGallery() {
    if (analyzing) return;
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 0.85,
        allowsEditing: true,
        aspect: [1, 1],
      });
      if (result.canceled || !result.assets[0]?.uri) return;
      setError(null);
      setAnalyzing(true);
      Animated.timing(overlayOpacity, { toValue: 1, duration: 300, useNativeDriver: true }).start();
      await navigateToResult(result.assets[0].uri);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Error al abrir galería';
      setError(msg);
    }
  }

  async function navigateToResult(uri: string) {
    try {
      const prediction = await predictLeaf(uri);
      try { await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success); } catch (_) {}
      navigation.navigate('Result', {
        predictionClass: prediction.raw_predicted_class,
        confidence:      prediction.confidence,
        isUncertain:     prediction.is_uncertain,
      });
    } finally {
      setAnalyzing(false);
      Animated.timing(overlayOpacity, { toValue: 0, duration: 300, useNativeDriver: true }).start();
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <View style={styles.root}>
      {/* Cámara */}
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFillObject} facing={facing} />

      {/* Overlay oscuro con recorte central */}
      <View style={styles.overlay} pointerEvents="none">
        {/* Banda superior */}
        <View style={[styles.overlayBand, { flex: 1 }]} />

        {/* Fila central: banda izquierda + marco + banda derecha */}
        <View style={styles.overlayRow}>
          <View style={[styles.overlayBand, { flex: 1 }]} />

          {/* Marco de escaneo */}
          <View style={[styles.frame, { width: FRAME_SIZE, height: FRAME_SIZE }]}>
            {/* Esquinas */}
            <View style={[styles.corner, styles.cornerTL]} />
            <View style={[styles.corner, styles.cornerTR]} />
            <View style={[styles.corner, styles.cornerBL]} />
            <View style={[styles.corner, styles.cornerBR]} />

            {/* Línea de escaneo */}
            <Animated.View
              style={[
                styles.scanLine,
                {
                  transform: [{
                    translateY: scanLineY.interpolate({
                      inputRange: [0, 1],
                      outputRange: [0, FRAME_SIZE - 2],
                    }),
                  }],
                },
              ]}
            />
          </View>

          <View style={[styles.overlayBand, { flex: 1 }]} />
        </View>

        {/* Banda inferior */}
        <View style={[styles.overlayBand, { flex: 1 }]} />
      </View>

      {/* Cabecera */}
      <SafeAreaView style={styles.headerArea} edges={['top']}>
        <Text style={styles.headerTitle}>DeepVine</Text>
        <Text style={styles.headerSub}>
          {analyzing ? 'Analizando hoja...' : 'Apunta a la hoja de vid'}
        </Text>
        {error && (
          <View style={styles.errorPill}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
      </SafeAreaView>

      {/* Controles inferiores */}
      <SafeAreaView style={styles.controls} edges={['bottom']}>
        {/* Galería */}
        <TouchableOpacity
          style={styles.ctrlBtn}
          onPress={handleGallery}
          disabled={analyzing}
        >
          <Ionicons name="images-outline" size={26} color={COLORS.textPrimary} />
          <Text style={styles.ctrlLabel}>Galería</Text>
        </TouchableOpacity>

        {/* Captura */}
        <Animated.View style={{ transform: [{ scale: captureScale }] }}>
          <TouchableOpacity
            style={[styles.captureBtn, analyzing && styles.captureBtnDisabled]}
            onPress={handleCapture}
            disabled={analyzing}
            activeOpacity={0.8}
          >
            {analyzing
              ? <ActivityIndicator color={COLORS.gold} size="large" />
              : <View style={styles.captureInner} />
            }
          </TouchableOpacity>
        </Animated.View>

        {/* Girar cámara */}
        <TouchableOpacity
          style={styles.ctrlBtn}
          onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
          disabled={analyzing}
        >
          <Ionicons name="camera-reverse-outline" size={26} color={COLORS.textPrimary} />
          <Text style={styles.ctrlLabel}>Voltear</Text>
        </TouchableOpacity>
      </SafeAreaView>

      {/* Overlay de análisis */}
      <Animated.View
        style={[styles.analyzingOverlay, { opacity: overlayOpacity }]}
        pointerEvents={analyzing ? 'auto' : 'none'}
      >
        <View style={styles.analyzingCard}>
          <ActivityIndicator color={COLORS.gold} size="large" />
          <Text style={styles.analyzingTitle}>Analizando</Text>
          <Text style={styles.analyzingBody}>
            La IA está identificando{'\n'}la variedad de vid…
          </Text>
        </View>
      </Animated.View>
    </View>
  );
}

// ─── Estilos ─────────────────────────────────────────────────────────────────

const CORNER_SIZE = CORNER;

const styles = StyleSheet.create({
  root:   { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: COLORS.background },

  // Overlay de oscurecimiento
  overlay: { ...StyleSheet.absoluteFillObject },
  overlayBand: { backgroundColor: 'rgba(0,0,0,0.62)' },
  overlayRow: { flexDirection: 'row' },

  // Marco
  frame: { position: 'relative' },
  corner: {
    position: 'absolute',
    width:    CORNER_SIZE,
    height:   CORNER_SIZE,
    borderColor: COLORS.gold,
  },
  cornerTL: { top: 0, left: 0, borderTopWidth: 3, borderLeftWidth: 3 },
  cornerTR: { top: 0, right: 0, borderTopWidth: 3, borderRightWidth: 3 },
  cornerBL: { bottom: 0, left: 0, borderBottomWidth: 3, borderLeftWidth: 3 },
  cornerBR: { bottom: 0, right: 0, borderBottomWidth: 3, borderRightWidth: 3 },
  scanLine: {
    position:        'absolute',
    left:            0,
    right:           0,
    height:          2,
    backgroundColor: COLORS.gold,
    opacity:         0.85,
    borderRadius:    1,
    shadowColor:     COLORS.gold,
    shadowOffset:    { width: 0, height: 0 },
    shadowOpacity:   1,
    shadowRadius:    6,
  },

  // Cabecera
  headerArea: {
    position: 'absolute',
    top: 0, left: 0, right: 0,
    alignItems: 'center',
    paddingTop: SPACING.sm,
  },
  headerTitle: {
    ...FONT.title,
    color: COLORS.gold,
    letterSpacing: 3,
    textShadowColor: 'rgba(0,0,0,0.8)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  headerSub: {
    ...FONT.caption,
    color: 'rgba(255,255,255,0.75)',
    marginTop: 4,
    textShadowColor: 'rgba(0,0,0,0.9)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  errorPill: {
    marginTop:        SPACING.sm,
    backgroundColor:  'rgba(229,115,115,0.9)',
    borderRadius:     RADIUS.full,
    paddingHorizontal: SPACING.md,
    paddingVertical:  SPACING.xs,
  },
  errorText: { ...FONT.caption, color: '#fff' },

  // Controles inferiores
  controls: {
    position:       'absolute',
    bottom:         0, left: 0, right: 0,
    flexDirection:  'row',
    alignItems:     'center',
    justifyContent: 'space-around',
    paddingVertical: SPACING.lg,
    paddingHorizontal: SPACING.xl,
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  ctrlBtn:   { alignItems: 'center', gap: 4, minWidth: 64 },
  ctrlLabel: { ...FONT.label, color: 'rgba(255,255,255,0.75)' },

  captureBtn: {
    width:           76,
    height:          76,
    borderRadius:    38,
    borderWidth:     4,
    borderColor:     '#fff',
    backgroundColor: 'rgba(255,255,255,0.15)',
    alignItems:      'center',
    justifyContent:  'center',
  },
  captureBtnDisabled: { borderColor: COLORS.gold, backgroundColor: 'rgba(201,168,76,0.15)' },
  captureInner: {
    width:           54,
    height:          54,
    borderRadius:    27,
    backgroundColor: '#fff',
  },

  // Permiso
  permissionContainer: {
    flex: 1,
    backgroundColor: COLORS.background,
    alignItems:     'center',
    justifyContent: 'center',
    padding:        SPACING.xl,
  },
  permissionIcon:  { fontSize: 64, marginBottom: SPACING.lg },
  permissionTitle: { ...FONT.title, marginBottom: SPACING.sm, textAlign: 'center' },
  permissionBody:  { ...FONT.body, color: COLORS.textSecondary, textAlign: 'center', marginBottom: SPACING.xl },
  permissionBtn: {
    backgroundColor: COLORS.primary,
    borderRadius:    RADIUS.lg,
    paddingHorizontal: SPACING.xl,
    paddingVertical:  SPACING.md,
  },
  permissionBtnText: { ...FONT.heading, color: COLORS.gold, fontSize: 15 },

  // Overlay analizando
  analyzingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(12,6,16,0.88)',
    alignItems:     'center',
    justifyContent: 'center',
  },
  analyzingCard: {
    backgroundColor: COLORS.surfaceElevated,
    borderRadius:    RADIUS.xl,
    borderWidth:     1,
    borderColor:     COLORS.border,
    padding:         SPACING.xl,
    alignItems:     'center',
    gap:             SPACING.md,
    width:           260,
  },
  analyzingTitle: { ...FONT.title, color: COLORS.gold },
  analyzingBody: {
    ...FONT.body,
    color:     COLORS.textSecondary,
    textAlign: 'center',
  },
});
