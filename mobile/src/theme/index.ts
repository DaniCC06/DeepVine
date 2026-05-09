export const COLORS = {
  // Backgrounds
  background:       '#0C0610',
  surface:          '#190D1A',
  surfaceElevated:  '#261525',
  surfaceHigh:      '#341B2E',

  // Brand
  primary:          '#821F3E',
  primaryDark:      '#5A0E25',
  primaryDeep:      '#3D0818',

  // Gold accent
  gold:             '#C9A84C',
  goldLight:        '#E8C870',
  goldDim:          '#8A6E30',

  // Text
  textPrimary:      '#F0E6D2',
  textSecondary:    '#B09080',
  textMuted:        '#6B4E4E',
  textDim:          '#3D2838',

  // Borders
  border:           '#3D2238',
  borderLight:      '#5A3050',

  // Status
  success:          '#81C784',
  error:            '#E57373',
  warning:          '#FFB74D',
} as const;

export const SPACING = {
  xs:  4,
  sm:  8,
  md:  16,
  lg:  24,
  xl:  32,
  xxl: 48,
} as const;

export const RADIUS = {
  sm:   8,
  md:   14,
  lg:   20,
  xl:   28,
  full: 9999,
} as const;

export const FONT = {
  titleLarge: {
    fontSize: 32,
    fontWeight: '800' as const,
    letterSpacing: 2,
    color: COLORS.textPrimary,
  },
  title: {
    fontSize: 22,
    fontWeight: '700' as const,
    letterSpacing: 0.8,
    color: COLORS.textPrimary,
  },
  heading: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: COLORS.textPrimary,
  },
  body: {
    fontSize: 15,
    fontWeight: '400' as const,
    color: COLORS.textPrimary,
    lineHeight: 22,
  },
  caption: {
    fontSize: 12,
    fontWeight: '500' as const,
    color: COLORS.textSecondary,
    letterSpacing: 0.3,
  },
  label: {
    fontSize: 10,
    fontWeight: '700' as const,
    letterSpacing: 1.5,
    color: COLORS.textMuted,
  },
} as const;
