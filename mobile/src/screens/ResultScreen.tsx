import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity, StyleSheet,
  Animated, Dimensions, Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as Haptics from 'expo-haptics';

import { COLORS, SPACING, RADIUS, FONT } from '../theme';
import { getVine, type WineInfo } from '../data/vineDatabase';
import { useCollection } from '../hooks/useCollection';
import type { RootStackParamList } from '../navigation';

const { width } = Dimensions.get('window');

type RoutePropT = RouteProp<RootStackParamList, 'Result'>;
type NavProp    = NativeStackNavigationProp<RootStackParamList>;

// ─── Componente métrica individual ───────────────────────────────────────────

interface MetricProps {
  icon: string;
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
  delay: number;
  masterAnim: Animated.Value;
}

function MetricTile({ icon, label, value, sub, accent, delay, masterAnim }: MetricProps) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    const timer = setTimeout(() => {
      Animated.spring(anim, { toValue: 1, friction: 8, tension: 60, useNativeDriver: true }).start();
    }, delay);
    return () => clearTimeout(timer);
  }, []);

  return (
    <Animated.View
      style={[
        styles.metricTile,
        accent && styles.metricTileAccent,
        {
          opacity: anim,
          transform: [{ scale: anim.interpolate({ inputRange: [0, 1], outputRange: [0.85, 1] }) }],
        },
      ]}
    >
      <Text style={styles.metricIcon}>{icon}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={[styles.metricValue, accent && styles.metricValueAccent]}>{value}</Text>
      {sub && <Text style={styles.metricSub}>{sub}</Text>}
    </Animated.View>
  );
}

// ─── Tarjeta de vino ─────────────────────────────────────────────────────────

function WineCard({ wine, index }: { wine: WineInfo; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const anim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const timer = setTimeout(() => {
      Animated.spring(anim, { toValue: 1, friction: 8, tension: 55, useNativeDriver: true }).start();
    }, 600 + index * 150);
    return () => clearTimeout(timer);
  }, []);

  const stars = Math.round(wine.balanceScore);

  return (
    <Animated.View
      style={[
        styles.wineCard,
        {
          opacity: anim,
          transform: [{ translateX: anim.interpolate({ inputRange: [0, 1], outputRange: [40, 0] }) }],
        },
      ]}
    >
      <TouchableOpacity activeOpacity={0.88} onPress={() => setExpanded(e => !e)}>
        {/* Cabecera del vino */}
        <View style={styles.wineCardHeader}>
          <View style={styles.scoreCircle}>
            <Text style={styles.scoreNum}>{wine.criticScore}</Text>
            <Text style={styles.scoreLabel}>pts</Text>
          </View>
          <View style={styles.wineCardInfo}>
            <Text style={styles.wineName} numberOfLines={1}>{wine.name}</Text>
            <Text style={styles.wineWinery}>{wine.winery}</Text>
            <Text style={styles.wineDenom}>{wine.denomination}</Text>
          </View>
          <Ionicons
            name={expanded ? 'chevron-up' : 'chevron-down'}
            size={18}
            color={COLORS.textMuted}
          />
        </View>

        {/* Detalles expandibles */}
        {expanded && (
          <View style={styles.wineDetails}>
            <View style={styles.wineDetailRow}>
              <Text style={styles.wineDetailIcon}>📍</Text>
              <Text style={styles.wineDetailText}>{wine.region}</Text>
            </View>
            <View style={styles.wineDetailRow}>
              <Text style={styles.wineDetailIcon}>🗓</Text>
              <Text style={styles.wineDetailText}>Añada {wine.vintage}</Text>
            </View>
            <View style={styles.wineDetailRow}>
              <Text style={styles.wineDetailIcon}>🍶</Text>
              <Text style={styles.wineDetailText}>{wine.alcoholDegrees}% vol</Text>
            </View>
            <View style={styles.wineDetailRow}>
              <Text style={styles.wineDetailIcon}>💶</Text>
              <Text style={styles.wineDetailText}>Precio medio: €{wine.avgPriceEur.toFixed(2)}</Text>
            </View>
            <View style={styles.wineDetailRow}>
              <Text style={styles.wineDetailIcon}>🧪</Text>
              <Text style={styles.wineDetailText}>pH {wine.ph.toFixed(2)} · Acidez {wine.totalAcidityGl.toFixed(1)} g/L</Text>
            </View>
            <View style={styles.wineDetailRow}>
              <Text style={styles.wineDetailIcon}>🪵</Text>
              <Text style={styles.wineDetailText}>{wine.aging}</Text>
            </View>
            <View style={styles.wineDetailRow}>
              <Text style={styles.wineDetailIcon}>⚖️</Text>
              <View>
                <Text style={styles.wineDetailText}>Equilibrio</Text>
                <View style={styles.starsRow}>
                  {[1, 2, 3, 4, 5].map(s => (
                    <Text key={s} style={styles.star}>
                      {s <= stars ? '★' : '☆'}
                    </Text>
                  ))}
                </View>
              </View>
            </View>
          </View>
        )}
      </TouchableOpacity>
    </Animated.View>
  );
}

// ─── Pantalla principal ───────────────────────────────────────────────────────

export default function ResultScreen() {
  const navigation = useNavigation<NavProp>();
  const route      = useRoute<RoutePropT>();
  const { predictionClass, confidence, isUncertain } = route.params;

  const { addDiscovery, isDiscovered } = useCollection();
  const vine = getVine(predictionClass);

  const [isNew, setIsNew] = useState(false);
  const [saved, setSaved] = useState(false);

  const headerAnim   = useRef(new Animated.Value(0)).current;
  const badgeScale   = useRef(new Animated.Value(0)).current;
  const badgeOpacity = useRef(new Animated.Value(0)).current;
  const shimmer      = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Entrada del header
    Animated.timing(headerAnim, { toValue: 1, duration: 600, useNativeDriver: true }).start();

    // Registrar en colección
    (async () => {
      if (vine && !isUncertain) {
        const newDisc = await addDiscovery(vine.id);
        if (newDisc) {
          setIsNew(true);
          try { await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success); } catch (_) {}

          // Animar badge de nueva variedad
          setTimeout(() => {
            Animated.parallel([
              Animated.spring(badgeScale, { toValue: 1, friction: 5, tension: 80, useNativeDriver: true }),
              Animated.timing(badgeOpacity, { toValue: 1, duration: 300, useNativeDriver: true }),
            ]).start();

            // Shimmer loop en el título
            Animated.loop(
              Animated.sequence([
                Animated.timing(shimmer, { toValue: 1, duration: 1000, useNativeDriver: true }),
                Animated.timing(shimmer, { toValue: 0, duration: 1000, useNativeDriver: true }),
              ]),
            ).start();
          }, 400);
        } else {
          setSaved(true);
        }
      }
    })();
  }, []);

  const confPct    = Math.round(confidence * 100);
  const confColor  = confidence >= 0.85 ? COLORS.success : confidence >= 0.7 ? COLORS.warning : COLORS.error;

  return (
    <View style={styles.root}>
      <LinearGradient colors={['#0C0610', '#190D1A']} style={StyleSheet.absoluteFillObject} />

      <SafeAreaView style={styles.safe} edges={['top']}>
        {/* ── Barra superior ────────────────────────────────── */}
        <View style={styles.topBar}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
            <Ionicons name="chevron-down" size={24} color={COLORS.textPrimary} />
          </TouchableOpacity>
          <Text style={styles.topBarTitle}>Resultado</Text>
          <View style={{ width: 40 }} />
        </View>

        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}
        >
          {/* ── Badge de nueva variedad ───────────────────── */}
          {isNew && (
            <Animated.View
              style={[
                styles.newBadge,
                {
                  opacity: badgeOpacity,
                  transform: [{ scale: badgeScale }],
                },
              ]}
            >
              <LinearGradient
                colors={[COLORS.gold, '#A07830', COLORS.gold]}
                start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}
                style={styles.newBadgeGrad}
              >
                <Text style={styles.newBadgeIcon}>✨</Text>
                <Text style={styles.newBadgeText}>¡NUEVA VARIEDAD DESCUBIERTA!</Text>
                <Text style={styles.newBadgeIcon}>✨</Text>
              </LinearGradient>
            </Animated.View>
          )}

          {/* ── Cabecera de la variedad ───────────────────── */}
          <Animated.View
            style={[
              styles.vineHeader,
              {
                opacity: headerAnim,
                transform: [{ translateY: headerAnim.interpolate({ inputRange: [0, 1], outputRange: [30, 0] }) }],
              },
            ]}
          >
            {isUncertain || !vine ? (
              <>
                <Text style={styles.vineEmoji}>🌿</Text>
                <Text style={styles.vineName}>Variedad Incierta</Text>
                <Text style={styles.vineLatinName}>
                  Confianza insuficiente para identificar la variedad.{'\n'}
                  Intenta con otra fotografía con mejor iluminación.
                </Text>
              </>
            ) : (
              <>
                <Animated.Text
                  style={[
                    styles.vineEmoji,
                    {
                      opacity: shimmer.interpolate({ inputRange: [0, 1], outputRange: [1, 0.6] }),
                    },
                  ]}
                >
                  {vine.emoji}
                </Animated.Text>
                <Animated.Text
                  style={[
                    styles.vineName,
                    {
                      color: shimmer.interpolate({
                        inputRange: [0, 1],
                        outputRange: [COLORS.textPrimary, COLORS.gold],
                      }),
                    },
                  ]}
                >
                  {vine.name}
                </Animated.Text>
                <Text style={styles.vineLatinName}>{vine.latinName}</Text>
                <Text style={styles.vineOrigin}>
                  <Ionicons name="location-outline" size={12} /> {vine.origin}
                </Text>

                {/* Rareza */}
                <View style={[
                  styles.rarityBadge,
                  vine.rarity === 'Rara' && styles.rarityRare,
                  vine.rarity === 'Poco común' && styles.rarityUncommon,
                ]}>
                  <Text style={styles.rarityText}>{vine.rarity}</Text>
                </View>
              </>
            )}

            {/* Barra de confianza */}
            <View style={styles.confBox}>
              <View style={styles.confHeader}>
                <Text style={styles.confLabel}>CONFIANZA DEL MODELO</Text>
                <Text style={[styles.confPct, { color: confColor }]}>{confPct}%</Text>
              </View>
              <View style={styles.confTrack}>
                <View style={[styles.confFill, { width: `${confPct}%`, backgroundColor: confColor }]} />
              </View>
            </View>
          </Animated.View>

          {/* ── Descripción ───────────────────────────────── */}
          {vine && !isUncertain && (
            <>
              <View style={styles.descCard}>
                <Text style={styles.descTitle}>Sobre esta variedad</Text>
                <Text style={styles.descBody}>{vine.description}</Text>

                <View style={styles.chipsRow}>
                  {vine.characteristics.map((c) => (
                    <View key={c} style={styles.chip}>
                      <Text style={styles.chipText}>{c}</Text>
                    </View>
                  ))}
                </View>
              </View>

              {/* ── Métricas del primer vino ─────────────── */}
              {vine.wines[0] && (
                <>
                  <Text style={styles.sectionLabel}>FICHA ANALÍTICA · {vine.wines[0].name.toUpperCase()}</Text>
                  <View style={styles.metricsGrid}>
                    <MetricTile icon="🧪" label="pH"           value={vine.wines[0].ph.toFixed(2)}            delay={0}   masterAnim={headerAnim} />
                    <MetricTile icon="⚗️" label="Acidez total" value={`${vine.wines[0].totalAcidityGl.toFixed(1)} g/L`} delay={80}  masterAnim={headerAnim} />
                    <MetricTile icon="🍶" label="Alcohol"      value={`${vine.wines[0].alcoholDegrees}°`}       delay={160} masterAnim={headerAnim} />
                    <MetricTile icon="🗓" label="Añada"        value={`${vine.wines[0].vintage}`}               delay={240} masterAnim={headerAnim} />
                    <MetricTile icon="🏆" label="Puntuación"   value={`${vine.wines[0].criticScore}/100`}        delay={320} masterAnim={headerAnim} accent />
                    <MetricTile icon="💶" label="Precio medio" value={`€${vine.wines[0].avgPriceEur.toFixed(0)}`} delay={400} masterAnim={headerAnim} />
                  </View>
                </>
              )}

              {/* ── Vinos recomendados ───────────────────── */}
              <Text style={styles.sectionLabel}>VINOS RECOMENDADOS</Text>
              {vine.wines.map((w, i) => (
                <WineCard key={w.name} wine={w} index={i} />
              ))}
            </>
          )}

          {/* ── Acciones ─────────────────────────────────── */}
          <View style={styles.actions}>
            <TouchableOpacity
              style={styles.actionSecondary}
              onPress={() => navigation.goBack()}
            >
              <Ionicons name="scan-outline" size={18} color={COLORS.gold} />
              <Text style={styles.actionSecondaryText}>Escanear otra hoja</Text>
            </TouchableOpacity>
          </View>

          <View style={{ height: SPACING.xxl }} />
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

// ─── Estilos ─────────────────────────────────────────────────────────────────

const TILE_SIZE = (width - SPACING.lg * 2 - SPACING.sm * 2) / 3;

const styles = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1 },
  scroll: { paddingHorizontal: SPACING.lg },

  // Top bar
  topBar: {
    flexDirection:  'row',
    alignItems:     'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingVertical:   SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  backBtn:     { padding: SPACING.xs },
  topBarTitle: { ...FONT.heading, fontSize: 16 },

  // Badge nueva variedad
  newBadge: { marginTop: SPACING.lg, borderRadius: RADIUS.lg, overflow: 'hidden' },
  newBadgeGrad: {
    flexDirection:    'row',
    alignItems:       'center',
    justifyContent:   'center',
    paddingVertical:  SPACING.sm,
    paddingHorizontal: SPACING.md,
    gap: SPACING.sm,
  },
  newBadgeText: {
    ...FONT.label,
    color:       COLORS.primaryDeep,
    fontSize:    11,
    fontWeight:  '800',
    letterSpacing: 1.5,
  },
  newBadgeIcon: { fontSize: 16 },

  // Cabecera variedad
  vineHeader: {
    alignItems:      'center',
    paddingVertical: SPACING.xl,
    gap:             SPACING.xs,
  },
  vineEmoji:     { fontSize: 72, marginBottom: SPACING.sm },
  vineName:      { ...FONT.titleLarge, fontSize: 28, textAlign: 'center' },
  vineLatinName: {
    ...FONT.caption,
    fontStyle:  'italic',
    textAlign:  'center',
    color:      COLORS.textSecondary,
    marginTop:  SPACING.xs,
  },
  vineOrigin: { ...FONT.caption, color: COLORS.textMuted, marginTop: 2 },

  rarityBadge: {
    marginTop:        SPACING.sm,
    paddingHorizontal: SPACING.md,
    paddingVertical:  4,
    borderRadius:     RADIUS.full,
    backgroundColor:  COLORS.surfaceElevated,
    borderWidth:      1,
    borderColor:      COLORS.border,
  },
  rarityRare:     { borderColor: COLORS.gold, backgroundColor: 'rgba(201,168,76,0.12)' },
  rarityUncommon: { borderColor: COLORS.borderLight, backgroundColor: COLORS.surfaceHigh },
  rarityText:     { ...FONT.label, color: COLORS.textSecondary },

  // Confianza
  confBox: {
    width:         '100%',
    marginTop:     SPACING.lg,
    gap:           SPACING.xs,
  },
  confHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  confLabel:  { ...FONT.label },
  confPct:    { ...FONT.heading, fontSize: 18, fontWeight: '800' },
  confTrack: {
    height:          8,
    backgroundColor: COLORS.surfaceElevated,
    borderRadius:    RADIUS.full,
    overflow:        'hidden',
  },
  confFill: { height: '100%', borderRadius: RADIUS.full },

  // Descripción
  descCard: {
    backgroundColor: COLORS.surfaceElevated,
    borderRadius:    RADIUS.lg,
    borderWidth:     1,
    borderColor:     COLORS.border,
    padding:         SPACING.lg,
    marginBottom:    SPACING.lg,
    gap:             SPACING.md,
  },
  descTitle: { ...FONT.heading },
  descBody:  { ...FONT.body, color: COLORS.textSecondary },
  chipsRow:  { flexDirection: 'row', flexWrap: 'wrap', gap: SPACING.xs },
  chip: {
    backgroundColor:  COLORS.surface,
    borderRadius:     RADIUS.full,
    paddingHorizontal: SPACING.sm,
    paddingVertical:  4,
    borderWidth:      1,
    borderColor:      COLORS.border,
  },
  chipText: { ...FONT.caption, color: COLORS.textSecondary, fontSize: 11 },

  // Métricas
  sectionLabel: {
    ...FONT.label,
    marginBottom: SPACING.sm,
    marginTop:    SPACING.lg,
    color:        COLORS.textMuted,
  },
  metricsGrid: {
    flexDirection:  'row',
    flexWrap:       'wrap',
    gap:            SPACING.sm,
    marginBottom:   SPACING.lg,
  },
  metricTile: {
    width:           TILE_SIZE,
    backgroundColor: COLORS.surfaceElevated,
    borderRadius:    RADIUS.md,
    borderWidth:     1,
    borderColor:     COLORS.border,
    padding:         SPACING.md,
    alignItems:      'center',
    gap:             3,
  },
  metricTileAccent: { borderColor: COLORS.goldDim, backgroundColor: 'rgba(201,168,76,0.08)' },
  metricIcon:       { fontSize: 22 },
  metricLabel:      { ...FONT.label, textAlign: 'center' },
  metricValue:      { ...FONT.heading, fontSize: 15, textAlign: 'center' },
  metricValueAccent: { color: COLORS.gold },
  metricSub:        { ...FONT.caption, textAlign: 'center', fontSize: 10 },

  // Vino card
  wineCard: {
    backgroundColor: COLORS.surfaceElevated,
    borderRadius:    RADIUS.lg,
    borderWidth:     1,
    borderColor:     COLORS.border,
    marginBottom:    SPACING.sm,
    overflow:        'hidden',
  },
  wineCardHeader: {
    flexDirection:  'row',
    alignItems:     'center',
    padding:        SPACING.md,
    gap:            SPACING.md,
  },
  scoreCircle: {
    width:           52,
    height:          52,
    borderRadius:    26,
    backgroundColor: COLORS.primaryDark,
    borderWidth:     2,
    borderColor:     COLORS.primary,
    alignItems:      'center',
    justifyContent:  'center',
  },
  scoreNum:   { ...FONT.heading, fontSize: 16, color: COLORS.gold, lineHeight: 18 },
  scoreLabel: { ...FONT.label, fontSize: 9, color: COLORS.goldDim },
  wineCardInfo: { flex: 1 },
  wineName:   { ...FONT.body, fontWeight: '700' },
  wineWinery: { ...FONT.caption, color: COLORS.textSecondary },
  wineDenom:  { ...FONT.label, fontSize: 9, color: COLORS.gold, marginTop: 2 },

  // Detalles vino
  wineDetails: {
    paddingHorizontal: SPACING.lg,
    paddingBottom:     SPACING.md,
    borderTopWidth:    1,
    borderTopColor:    COLORS.border,
    gap:               SPACING.sm,
  },
  wineDetailRow: { flexDirection: 'row', gap: SPACING.sm, alignItems: 'flex-start' },
  wineDetailIcon: { fontSize: 14, marginTop: 1 },
  wineDetailText: { ...FONT.body, color: COLORS.textSecondary, fontSize: 13, flex: 1 },
  starsRow:  { flexDirection: 'row', marginTop: 2 },
  star:      { color: COLORS.gold, fontSize: 14, marginRight: 1 },

  // Acciones
  actions: {
    marginTop:  SPACING.xl,
    gap:        SPACING.md,
    alignItems: 'center',
  },
  actionSecondary: {
    flexDirection:    'row',
    alignItems:       'center',
    gap:              SPACING.sm,
    paddingHorizontal: SPACING.xl,
    paddingVertical:  SPACING.md,
    borderRadius:     RADIUS.lg,
    borderWidth:      1,
    borderColor:      COLORS.gold,
  },
  actionSecondaryText: { ...FONT.body, color: COLORS.gold, fontWeight: '600' },
});
