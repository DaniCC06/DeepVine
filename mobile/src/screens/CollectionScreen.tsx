import React, { useEffect, useRef, useCallback } from 'react';
import {
  View, Text, FlatList, TouchableOpacity, StyleSheet,
  Animated, Dimensions, RefreshControl,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

import { COLORS, SPACING, RADIUS, FONT } from '../theme';
import { VINE_DATABASE, ALL_VINE_IDS, type VineEntry } from '../data/vineDatabase';
import { useCollection } from '../hooks/useCollection';

const { width } = Dimensions.get('window');
const COLS      = 2;
const CARD_W    = (width - SPACING.lg * 2 - SPACING.sm) / COLS;
const CARD_H    = CARD_W * 1.25;

// ─── Tarjeta individual ───────────────────────────────────────────────────────

interface CardProps {
  vine:        VineEntry;
  discovered:  boolean;
  scansCount:  number;
  discoveredAt: string | null;
  index:       number;
}

function VineCard({ vine, discovered, scansCount, discoveredAt, index }: CardProps) {
  const anim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const timer = setTimeout(() => {
      Animated.spring(anim, { toValue: 1, friction: 8, tension: 60, useNativeDriver: true }).start();
    }, index * 90);
    return () => clearTimeout(timer);
  }, []);

  const discDate = discoveredAt
    ? new Date(discoveredAt).toLocaleDateString('es-ES', { day: '2-digit', month: 'short', year: 'numeric' })
    : null;

  return (
    <Animated.View
      style={[
        styles.cardWrap,
        {
          opacity: anim,
          transform: [
            { scale: anim.interpolate({ inputRange: [0, 1], outputRange: [0.8, 1] }) },
          ],
        },
      ]}
    >
      <LinearGradient
        colors={discovered ? vine.gradientColors : ['#180D1C', '#120A16', '#0D0811']}
        style={styles.card}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        {/* Decoración de fondo */}
        {!discovered && (
          <Text style={styles.cardBgEmoji}>🔒</Text>
        )}

        {/* Contenido */}
        <View style={styles.cardContent}>
          {discovered ? (
            <>
              <Text style={styles.cardEmoji}>{vine.emoji}</Text>
              <Text style={styles.cardName} numberOfLines={1}>{vine.name}</Text>
              <Text style={styles.cardLatin} numberOfLines={1}>{vine.latinName.split(' cv. ')[1] ?? ''}</Text>

              <View style={[
                styles.cardRarityBadge,
                vine.rarity === 'Rara' && styles.rarityRare,
                vine.rarity === 'Poco común' && styles.rarityUncommon,
              ]}>
                <Text style={styles.cardRarityText}>{vine.rarity.toUpperCase()}</Text>
              </View>
            </>
          ) : (
            <>
              <View style={styles.lockCircle}>
                <Ionicons name="lock-closed" size={28} color={COLORS.textDim} />
              </View>
              <Text style={styles.unknownName}>???</Text>
              <Text style={styles.unknownSub}>Por descubrir</Text>
            </>
          )}
        </View>

        {/* Footer */}
        <View style={styles.cardFooter}>
          {discovered ? (
            <>
              <View style={styles.discoveredDot} />
              <Text style={styles.cardDate} numberOfLines={1}>
                {discDate ?? ''}
              </Text>
              {scansCount > 1 && (
                <View style={styles.scansBadge}>
                  <Text style={styles.scansText}>{scansCount}×</Text>
                </View>
              )}
            </>
          ) : (
            <Text style={styles.lockHint}>Escanea una hoja</Text>
          )}
        </View>
      </LinearGradient>
    </Animated.View>
  );
}

// ─── Pantalla principal ───────────────────────────────────────────────────────

export default function CollectionScreen() {
  const { collection, total, loading, resetCollection } = useCollection();
  const discovered = collection.length;
  const headerAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(headerAnim, { toValue: 1, duration: 700, useNativeDriver: true }).start();
  }, []);

  const progressPct = total > 0 ? discovered / total : 0;

  const renderCard = useCallback(
    ({ item, index }: { item: string; index: number }) => {
      const vine    = VINE_DATABASE[item];
      const entry   = collection.find((c) => c.vineId === item);
      const isFound = Boolean(entry);
      return (
        <VineCard
          key={item}
          vine={vine}
          discovered={isFound}
          scansCount={entry?.scansCount ?? 0}
          discoveredAt={entry?.discoveredAt ?? null}
          index={index}
        />
      );
    },
    [collection],
  );

  return (
    <LinearGradient colors={['#0C0610', '#190D1A', '#220F1E']} style={styles.root}>
      <SafeAreaView style={styles.safe} edges={['top']}>

        {/* ── Cabecera ──────────────────────────────────── */}
        <Animated.View
          style={[
            styles.header,
            {
              opacity: headerAnim,
              transform: [{ translateY: headerAnim.interpolate({ inputRange: [0, 1], outputRange: [-20, 0] }) }],
            },
          ]}
        >
          <View>
            <Text style={styles.headerLabel}>MI BODEGA</Text>
            <Text style={styles.headerTitle}>Colección</Text>
          </View>
          <View style={styles.headerStats}>
            <Text style={styles.headerStatsNum}>
              <Text style={styles.headerHL}>{discovered}</Text>
              <Text style={styles.headerOf}>/{total}</Text>
            </Text>
            <Text style={styles.headerStatsLabel}>descubiertas</Text>
          </View>
        </Animated.View>

        {/* ── Barra de progreso ─────────────────────────── */}
        <Animated.View style={[styles.progressWrap, { opacity: headerAnim }]}>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${progressPct * 100}%` }]} />
          </View>
          {discovered === total && total > 0 ? (
            <Text style={styles.progressComplete}>✨ ¡Colección completa!</Text>
          ) : (
            <Text style={styles.progressHint}>
              {total - discovered} variedad{total - discovered !== 1 ? 'es' : ''} por descubrir
            </Text>
          )}
        </Animated.View>

        {/* ── Trofeos de logros ─────────────────────────── */}
        <View style={styles.achievementsRow}>
          {[
            { icon: '🥇', label: 'Primera',  unlocked: discovered >= 1 },
            { icon: '🍷', label: 'Sommelier', unlocked: discovered >= 2 },
            { icon: '👑', label: 'Maestro',   unlocked: discovered === total && total > 0 },
          ].map((a) => (
            <View key={a.label} style={[styles.achievement, !a.unlocked && styles.achievementLocked]}>
              <Text style={[styles.achievementIcon, !a.unlocked && styles.achievementIconDim]}>
                {a.unlocked ? a.icon : '🔒'}
              </Text>
              <Text style={[styles.achievementLabel, !a.unlocked && styles.achievementLabelDim]}>
                {a.label}
              </Text>
            </View>
          ))}
        </View>

        {/* ── Grid de variedades ────────────────────────── */}
        <FlatList
          data={ALL_VINE_IDS}
          keyExtractor={(id) => id}
          numColumns={COLS}
          renderItem={renderCard}
          contentContainerStyle={styles.grid}
          columnWrapperStyle={styles.columnWrapper}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={loading}
              tintColor={COLORS.gold}
              colors={[COLORS.gold]}
            />
          }
          ListFooterComponent={
            <View style={styles.footer}>
              <Text style={styles.footerText}>
                DeepVine · {discovered}/{total} variedades identificadas
              </Text>
            </View>
          }
        />
      </SafeAreaView>
    </LinearGradient>
  );
}

// ─── Estilos ─────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1 },

  // Header
  header: {
    flexDirection:    'row',
    justifyContent:   'space-between',
    alignItems:       'flex-end',
    paddingHorizontal: SPACING.lg,
    paddingTop:       SPACING.md,
    paddingBottom:    SPACING.md,
  },
  headerLabel:      { ...FONT.label, marginBottom: 2 },
  headerTitle:      { ...FONT.titleLarge, fontSize: 26 },
  headerStats:      { alignItems: 'flex-end' },
  headerStatsNum:   { fontSize: 28, fontWeight: '700' },
  headerHL:         { color: COLORS.gold, fontSize: 32, fontWeight: '800' },
  headerOf:         { color: COLORS.textMuted, fontSize: 20 },
  headerStatsLabel: { ...FONT.caption, color: COLORS.textMuted },

  // Progreso
  progressWrap: {
    paddingHorizontal: SPACING.lg,
    marginBottom:      SPACING.sm,
    gap:               SPACING.xs,
  },
  progressTrack: {
    height:           6,
    backgroundColor:  COLORS.surfaceElevated,
    borderRadius:     RADIUS.full,
    overflow:         'hidden',
    borderWidth:      1,
    borderColor:      COLORS.border,
  },
  progressFill: {
    height:           '100%',
    backgroundColor:  COLORS.gold,
    borderRadius:     RADIUS.full,
  },
  progressComplete: { ...FONT.caption, color: COLORS.gold, textAlign: 'right' },
  progressHint:     { ...FONT.caption, color: COLORS.textMuted, textAlign: 'right' },

  // Logros
  achievementsRow: {
    flexDirection:    'row',
    justifyContent:   'center',
    gap:              SPACING.sm,
    paddingHorizontal: SPACING.lg,
    paddingVertical:   SPACING.md,
  },
  achievement: {
    backgroundColor:  COLORS.surfaceElevated,
    borderRadius:     RADIUS.md,
    borderWidth:      1,
    borderColor:      COLORS.gold,
    paddingHorizontal: SPACING.md,
    paddingVertical:   SPACING.sm,
    alignItems:       'center',
    flex:             1,
    gap:              4,
  },
  achievementLocked: { borderColor: COLORS.border, opacity: 0.55 },
  achievementIcon:   { fontSize: 22 },
  achievementIconDim: { opacity: 0.4 },
  achievementLabel:  { ...FONT.label, fontSize: 9, color: COLORS.gold },
  achievementLabelDim: { color: COLORS.textDim },

  // Grid
  grid:          { paddingHorizontal: SPACING.lg, paddingBottom: SPACING.xxl },
  columnWrapper: { gap: SPACING.sm, marginBottom: SPACING.sm },

  // Tarjeta
  cardWrap: { width: CARD_W },
  card: {
    width:          '100%',
    height:         CARD_H,
    borderRadius:   RADIUS.lg,
    borderWidth:    1,
    borderColor:    COLORS.border,
    overflow:       'hidden',
    justifyContent: 'space-between',
    padding:        SPACING.md,
  },
  cardBgEmoji: {
    position:  'absolute',
    fontSize:  80,
    opacity:   0.06,
    bottom:    -10,
    right:     -10,
  },
  cardContent: {
    flex:       1,
    alignItems: 'center',
    justifyContent: 'center',
    gap:        6,
  },
  cardEmoji:   { fontSize: 42 },
  cardName: {
    ...FONT.body,
    fontWeight: '700',
    textAlign:  'center',
    fontSize:   15,
  },
  cardLatin: {
    ...FONT.caption,
    fontStyle:  'italic',
    textAlign:  'center',
    color:      COLORS.textSecondary,
    fontSize:   10,
  },
  cardRarityBadge: {
    marginTop:        4,
    paddingHorizontal: SPACING.sm,
    paddingVertical:  3,
    borderRadius:     RADIUS.full,
    backgroundColor:  'rgba(255,255,255,0.08)',
    borderWidth:      1,
    borderColor:      COLORS.border,
  },
  rarityRare:     { borderColor: COLORS.gold, backgroundColor: 'rgba(201,168,76,0.15)' },
  rarityUncommon: { borderColor: COLORS.borderLight },
  cardRarityText: { ...FONT.label, fontSize: 8 },

  lockCircle: {
    width:            64,
    height:           64,
    borderRadius:     32,
    backgroundColor:  'rgba(255,255,255,0.05)',
    borderWidth:      1,
    borderColor:      COLORS.textDim,
    alignItems:       'center',
    justifyContent:   'center',
  },
  unknownName: { ...FONT.heading, color: COLORS.textDim, letterSpacing: 3 },
  unknownSub:  { ...FONT.label, color: COLORS.textDim },

  cardFooter: {
    flexDirection:  'row',
    alignItems:     'center',
    gap:            SPACING.xs,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.06)',
    paddingTop:     SPACING.sm,
    marginTop:      SPACING.xs,
  },
  discoveredDot: {
    width:           5,
    height:          5,
    borderRadius:    2.5,
    backgroundColor: COLORS.success,
  },
  cardDate: { ...FONT.label, fontSize: 9, color: COLORS.textMuted, flex: 1 },
  scansBadge: {
    backgroundColor:  COLORS.primaryDark,
    borderRadius:     RADIUS.sm,
    paddingHorizontal: 5,
    paddingVertical:  2,
  },
  scansText: { ...FONT.label, fontSize: 9, color: COLORS.gold },
  lockHint:  { ...FONT.label, fontSize: 9, color: COLORS.textDim },

  // Footer
  footer: {
    paddingTop:  SPACING.xl,
    alignItems: 'center',
  },
  footerText: { ...FONT.label, color: COLORS.textDim },
});
