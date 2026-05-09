import React, { useEffect, useRef } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, Animated,
  ScrollView, Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';

import { COLORS, SPACING, RADIUS, FONT } from '../theme';
import { useCollection } from '../hooks/useCollection';
import { VINE_DATABASE } from '../data/vineDatabase';
import type { TabParamList } from '../navigation';

const { width } = Dimensions.get('window');
const CARD_SIZE = (width - SPACING.lg * 2 - SPACING.sm * 2) / 3;

type NavProp = BottomTabNavigationProp<TabParamList>;

export default function HomeScreen() {
  const navigation = useNavigation<NavProp>();
  const { collection, total } = useCollection();
  const discovered = collection.length;

  const fadeAnim  = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const glowAnim  = useRef(new Animated.Value(0.6)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 900, useNativeDriver: true }).start();

    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.055, duration: 1400, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1.0,   duration: 1400, useNativeDriver: true }),
      ]),
    );
    const glow = Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, { toValue: 1.0, duration: 1400, useNativeDriver: true }),
        Animated.timing(glowAnim, { toValue: 0.6, duration: 1400, useNativeDriver: true }),
      ]),
    );
    pulse.start();
    glow.start();
    return () => { pulse.stop(); glow.stop(); };
  }, []);

  const progressPct = total > 0 ? discovered / total : 0;
  const allVines    = Object.values(VINE_DATABASE);

  return (
    <LinearGradient colors={['#0C0610', '#190D1A', '#220F1E']} style={styles.root}>
      <SafeAreaView style={styles.safe} edges={['top']}>
        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}
        >
          {/* ── Logo / Hero ────────────────────────────────────── */}
          <Animated.View style={[styles.hero, { opacity: fadeAnim }]}>
            <Text style={styles.heroGrape}>🍇</Text>
            <Text style={styles.heroTitle}>DEEPVINE</Text>
            <Text style={styles.heroTagline}>
              Identifica variedades de vid{'\n'}mediante inteligencia artificial
            </Text>
          </Animated.View>

          {/* ── Progreso ──────────────────────────────────────── */}
          <Animated.View style={[styles.progressCard, { opacity: fadeAnim }]}>
            <View style={styles.progressRow}>
              <View>
                <Text style={styles.progressLabel}>MI BODEGA</Text>
                <Text style={styles.progressTitle}>
                  <Text style={styles.progressHL}>{discovered}</Text>
                  <Text style={styles.progressOf}> / {total} variedades</Text>
                </Text>
              </View>
              {discovered === total && total > 0 && (
                <View style={styles.completeBadge}>
                  <Text style={styles.completeBadgeText}>✨ Completa</Text>
                </View>
              )}
            </View>

            <View style={styles.progressBarTrack}>
              <Animated.View
                style={[
                  styles.progressBarFill,
                  { width: `${Math.max(progressPct * 100, 4)}%` },
                ]}
              />
            </View>

            <Text style={styles.progressHint}>
              {discovered === 0
                ? 'Escanea tu primera hoja para comenzar'
                : discovered < total
                ? `${total - discovered} variedad${total - discovered !== 1 ? 'es' : ''} por descubrir`
                : '¡Has completado la colección!'}
            </Text>
          </Animated.View>

          {/* ── Botón de escaneo ──────────────────────────────── */}
          <Animated.View
            style={[styles.scanWrap, { transform: [{ scale: pulseAnim }] }]}
          >
            <Animated.View
              style={[styles.scanGlow, { opacity: glowAnim }]}
            />
            <TouchableOpacity
              activeOpacity={0.88}
              onPress={() => navigation.navigate('Scan')}
              style={styles.scanBtn}
            >
              <LinearGradient
                colors={['#9B2347', '#6B1A35', '#3D0818']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.scanBtnInner}
              >
                <View style={styles.scanIconRing}>
                  <Ionicons name="scan" size={38} color={COLORS.gold} />
                </View>
                <Text style={styles.scanBtnTitle}>ESCANEAR HOJA</Text>
                <Text style={styles.scanBtnSub}>
                  Apunta la cámara a una hoja de vid
                </Text>
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>

          {/* ── Mini colección ───────────────────────────────── */}
          <Animated.View style={[styles.section, { opacity: fadeAnim }]}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionLabel}>VARIEDADES</Text>
              <TouchableOpacity onPress={() => navigation.navigate('Collection')}>
                <Text style={styles.sectionLink}>Ver todas →</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.miniGrid}>
              {allVines.map((vine) => {
                const found = collection.some((c) => c.vineId === vine.id);
                return (
                  <TouchableOpacity
                    key={vine.id}
                    onPress={() => navigation.navigate('Collection')}
                    activeOpacity={0.8}
                  >
                    <LinearGradient
                      colors={found ? vine.gradientColors : ['#1C1020', '#150C18', '#100A14']}
                      style={styles.miniCard}
                    >
                      <Text style={styles.miniEmoji}>{found ? vine.emoji : '🔒'}</Text>
                      <Text
                        style={[styles.miniName, !found && styles.miniNameHidden]}
                        numberOfLines={1}
                      >
                        {found ? vine.name : '???'}
                      </Text>
                      {found && (
                        <View style={styles.miniDiscoveredDot} />
                      )}
                      <Text style={[styles.miniRarity, !found && { color: COLORS.textDim }]}>
                        {found ? vine.rarity : '· · ·'}
                      </Text>
                    </LinearGradient>
                  </TouchableOpacity>
                );
              })}
            </View>
          </Animated.View>

          {/* ── Footer ───────────────────────────────────────── */}
          <Text style={styles.footer}>DeepVine · Clasificación IA de Variedades de Vid</Text>
        </ScrollView>
      </SafeAreaView>
    </LinearGradient>
  );
}

// ─── Estilos ─────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root:  { flex: 1 },
  safe:  { flex: 1 },
  scroll: { paddingHorizontal: SPACING.lg, paddingBottom: SPACING.xxl },

  // Hero
  hero:       { alignItems: 'center', paddingTop: SPACING.xl, paddingBottom: SPACING.lg },
  heroGrape:  { fontSize: 64, marginBottom: SPACING.sm },
  heroTitle:  { ...FONT.titleLarge, color: COLORS.gold, letterSpacing: 6, marginBottom: SPACING.sm },
  heroTagline: {
    ...FONT.caption,
    textAlign: 'center',
    lineHeight: 20,
    color: COLORS.textSecondary,
  },

  // Progreso
  progressCard: {
    backgroundColor: COLORS.surfaceElevated,
    borderRadius:    RADIUS.lg,
    borderWidth:     1,
    borderColor:     COLORS.border,
    padding:         SPACING.lg,
    marginBottom:    SPACING.xl,
  },
  progressRow:   { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: SPACING.md },
  progressLabel: { ...FONT.label, marginBottom: 4 },
  progressTitle: { fontSize: 20, fontWeight: '700', color: COLORS.textPrimary },
  progressHL:    { color: COLORS.gold, fontSize: 26, fontWeight: '800' },
  progressOf:    { color: COLORS.textSecondary, fontSize: 16 },
  progressBarTrack: {
    height:           6,
    backgroundColor:  COLORS.surface,
    borderRadius:     RADIUS.full,
    overflow:         'hidden',
    marginBottom:     SPACING.sm,
  },
  progressBarFill: {
    height:           '100%',
    backgroundColor:  COLORS.gold,
    borderRadius:     RADIUS.full,
  },
  progressHint: { ...FONT.caption, color: COLORS.textMuted },
  completeBadge: {
    backgroundColor: COLORS.goldDim,
    borderRadius:    RADIUS.sm,
    paddingHorizontal: SPACING.sm,
    paddingVertical:   4,
  },
  completeBadgeText: { ...FONT.label, color: COLORS.gold },

  // Botón scan
  scanWrap: { alignItems: 'center', marginBottom: SPACING.xl },
  scanGlow: {
    position:        'absolute',
    width:           width * 0.72,
    height:          130,
    backgroundColor: COLORS.primary,
    borderRadius:    RADIUS.xl,
    top:             10,
    opacity:         0.25,
    // blur simulado con sombra (Android no soporta shadowRadius igual que iOS)
  },
  scanBtn: {
    width:        width - SPACING.lg * 2,
    borderRadius: RADIUS.xl,
    overflow:     'hidden',
    borderWidth:  1,
    borderColor:  COLORS.primaryDark,
  },
  scanBtnInner: {
    alignItems:     'center',
    paddingVertical: SPACING.xl,
    paddingHorizontal: SPACING.lg,
  },
  scanIconRing: {
    width:            80,
    height:           80,
    borderRadius:     40,
    borderWidth:      2,
    borderColor:      COLORS.goldDim,
    backgroundColor:  'rgba(201,168,76,0.1)',
    alignItems:       'center',
    justifyContent:   'center',
    marginBottom:     SPACING.md,
  },
  scanBtnTitle: {
    ...FONT.titleLarge,
    fontSize:    20,
    color:       COLORS.gold,
    letterSpacing: 3,
    marginBottom: SPACING.xs,
  },
  scanBtnSub: { ...FONT.caption, color: COLORS.textSecondary },

  // Mini grid
  section:      { marginBottom: SPACING.xl },
  sectionHeader: {
    flexDirection:  'row',
    justifyContent: 'space-between',
    alignItems:     'center',
    marginBottom:   SPACING.md,
  },
  sectionLabel: { ...FONT.label, color: COLORS.textMuted },
  sectionLink:  { ...FONT.caption, color: COLORS.gold },
  miniGrid: {
    flexDirection:  'row',
    gap:            SPACING.sm,
  },
  miniCard: {
    width:          CARD_SIZE,
    borderRadius:   RADIUS.md,
    padding:        SPACING.sm,
    alignItems:     'center',
    borderWidth:    1,
    borderColor:    COLORS.border,
    minHeight:      110,
    justifyContent: 'center',
    gap:            4,
  },
  miniEmoji:         { fontSize: 28 },
  miniName:          { ...FONT.caption, fontSize: 11, textAlign: 'center', fontWeight: '600' },
  miniNameHidden:    { color: COLORS.textDim },
  miniDiscoveredDot: {
    width:           6,
    height:          6,
    borderRadius:    3,
    backgroundColor: COLORS.success,
  },
  miniRarity: { ...FONT.label, fontSize: 9, textAlign: 'center', color: COLORS.goldDim },

  // Footer
  footer: {
    textAlign: 'center',
    ...FONT.label,
    color:      COLORS.textDim,
    marginTop:  SPACING.md,
  },
});
