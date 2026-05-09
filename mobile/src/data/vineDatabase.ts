export interface WineInfo {
  name: string;
  winery: string;
  denomination: string;
  region: string;
  criticScore: number;
  avgPriceEur: number;
  ph: number;
  totalAcidityGl: number;
  aging: string;
  balanceScore: number;
  vintage: number;
  alcoholDegrees: number;
}

export interface VineEntry {
  id: string;
  name: string;
  latinName: string;
  origin: string;
  description: string;
  characteristics: string[];
  emoji: string;
  gradientColors: [string, string, string];
  cardColor: string;
  rarity: 'Común' | 'Poco común' | 'Rara';
  wines: WineInfo[];
}

export const VINE_DATABASE: Record<string, VineEntry> = {
  tipo1: {
    id: 'tipo1',
    name: 'Tempranillo',
    latinName: 'Vitis vinifera cv. Tempranillo',
    origin: 'Ribera del Duero, España',
    description:
      'La variedad tinta más emblemática de España. Protagonista indiscutible de los grandes Riojas y Riberas del Duero, su nombre proviene de que madura "temprano" respecto a otras variedades. Produce vinos complejos con aromas de fruta roja, cuero, tabaco y especias.',
    characteristics: ['Frutos rojos', 'Cuero', 'Tabaco', 'Vainilla', 'Especias'],
    emoji: '🍷',
    gradientColors: ['#6B1A2A', '#8B1F38', '#3D0C18'],
    cardColor: '#6B1A2A',
    rarity: 'Común',
    wines: [
      {
        name: 'Gran Reserva 904',
        winery: 'La Rioja Alta',
        denomination: 'DOCa Rioja',
        region: 'La Rioja, España',
        criticScore: 96,
        avgPriceEur: 42.0,
        ph: 3.48,
        totalAcidityGl: 5.5,
        aging: '36 meses en roble americano + 48 meses en botella',
        balanceScore: 5.0,
        vintage: 2016,
        alcoholDegrees: 13.5,
      },
      {
        name: 'Pesquera Reserva',
        winery: 'Alejandro Fernández',
        denomination: 'DO Ribera del Duero',
        region: 'Ribera del Duero, España',
        criticScore: 92,
        avgPriceEur: 28.5,
        ph: 3.52,
        totalAcidityGl: 5.8,
        aging: '18 meses en barrica de roble americano',
        balanceScore: 4.5,
        vintage: 2019,
        alcoholDegrees: 14.0,
      },
      {
        name: 'Cepa 21 Crianza',
        winery: 'Bodegas Cepa 21',
        denomination: 'DO Ribera del Duero',
        region: 'Ribera del Duero, España',
        criticScore: 89,
        avgPriceEur: 18.0,
        ph: 3.44,
        totalAcidityGl: 6.0,
        aging: '12 meses en roble francés y americano',
        balanceScore: 4.2,
        vintage: 2020,
        alcoholDegrees: 14.5,
      },
    ],
  },

  tipo2: {
    id: 'tipo2',
    name: 'Garnacha Tinta',
    latinName: 'Vitis vinifera cv. Grenache Noir',
    origin: 'Aragón, España',
    description:
      'Una de las uvas tintas más plantadas del mundo. Resistente a la sequía y al calor extremo, la Garnacha produce vinos generosos, de elevado alcohol y carácter herbáceo. En zonas de viñedo viejo da algunos de los vinos más concentrados y complejos de España.',
    characteristics: ['Frutos rojos maduros', 'Regaliz', 'Hierbas silvestres', 'Especias', 'Confitado'],
    emoji: '🍇',
    gradientColors: ['#5C1A4A', '#7A1F60', '#3D0C38'],
    cardColor: '#5C1A4A',
    rarity: 'Poco común',
    wines: [
      {
        name: 'Clos Erasmus',
        winery: 'Clos Erasmus',
        denomination: 'DOQ Priorat',
        region: 'Priorat, Cataluña',
        criticScore: 97,
        avgPriceEur: 110.0,
        ph: 3.68,
        totalAcidityGl: 4.9,
        aging: '18 meses en barricas de roble francés nuevo de 500L',
        balanceScore: 5.0,
        vintage: 2018,
        alcoholDegrees: 15.5,
      },
      {
        name: 'Las Rocas de San Alejandro',
        winery: 'Bodegas San Alejandro',
        denomination: 'DO Calatayud',
        region: 'Calatayud, Aragón',
        criticScore: 90,
        avgPriceEur: 14.0,
        ph: 3.72,
        totalAcidityGl: 4.6,
        aging: '8 meses en barrica de roble francés',
        balanceScore: 4.3,
        vintage: 2021,
        alcoholDegrees: 14.5,
      },
    ],
  },

  tipo3: {
    id: 'tipo3',
    name: 'Verdejo',
    latinName: 'Vitis vinifera cv. Verdejo',
    origin: 'Rueda, Castilla y León',
    description:
      'La joya blanca de Castilla. Variedad autóctona de la DO Rueda, produce vinos blancos aromáticos y frescos con notas cítricas, hierbas aromáticas y un característico amargor final. Resistente a la oxidación y de gran personalidad, es la blanca española por excelencia.',
    characteristics: ['Cítricos', 'Hierbas frescas', 'Hinojo', 'Melocotón blanco', 'Almendra'],
    emoji: '🍋',
    gradientColors: ['#4A6B1A', '#5E8A1F', '#2A3D0C'],
    cardColor: '#4A6B1A',
    rarity: 'Rara',
    wines: [
      {
        name: 'Belondrade y Lurton',
        winery: 'Belondrade',
        denomination: 'DO Rueda Superior',
        region: 'Rueda, Castilla y León',
        criticScore: 94,
        avgPriceEur: 38.0,
        ph: 3.22,
        totalAcidityGl: 6.8,
        aging: '10 meses en barrica de roble francés borgoñona',
        balanceScore: 4.8,
        vintage: 2021,
        alcoholDegrees: 13.5,
      },
      {
        name: 'Marqués de Riscal Verdejo',
        winery: 'Marqués de Riscal',
        denomination: 'DO Rueda',
        region: 'Rueda, Castilla y León',
        criticScore: 89,
        avgPriceEur: 11.5,
        ph: 3.18,
        totalAcidityGl: 6.5,
        aging: 'Sin crianza — fermentación fría a temperatura controlada',
        balanceScore: 4.4,
        vintage: 2022,
        alcoholDegrees: 13.0,
      },
      {
        name: 'Naia',
        winery: 'Bodegas Naia',
        denomination: 'DO Rueda',
        region: 'Rueda, Castilla y León',
        criticScore: 91,
        avgPriceEur: 14.0,
        ph: 3.25,
        totalAcidityGl: 6.3,
        aging: 'Sin crianza — vendimia nocturna',
        balanceScore: 4.6,
        vintage: 2022,
        alcoholDegrees: 13.0,
      },
    ],
  },
};

export const ALL_VINE_IDS = Object.keys(VINE_DATABASE);
export const TOTAL_VINES = ALL_VINE_IDS.length;

export function getVine(classId: string): VineEntry | null {
  return VINE_DATABASE[classId] ?? null;
}
