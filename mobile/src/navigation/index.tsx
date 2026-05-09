import React from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';

import { COLORS } from '../theme';
import HomeScreen from '../screens/HomeScreen';
import ScanScreen from '../screens/ScanScreen';
import ResultScreen from '../screens/ResultScreen';
import CollectionScreen from '../screens/CollectionScreen';

// ─── Tipos de navegación ─────────────────────────────────────────────────────

export type RootStackParamList = {
  Main: undefined;
  Result: {
    predictionClass: string;
    confidence: number;
    isUncertain: boolean;
  };
};

export type TabParamList = {
  Home: undefined;
  Scan: undefined;
  Collection: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<TabParamList>();

// ─── Tab bar personalizada ────────────────────────────────────────────────────

type TabIconName = keyof typeof Ionicons.glyphMap;

const TAB_CONFIG: Record<keyof TabParamList, { label: string; icon: TabIconName; iconActive: TabIconName }> = {
  Home:       { label: 'Inicio',    icon: 'home-outline',    iconActive: 'home' },
  Scan:       { label: 'Escanear',  icon: 'scan-outline',    iconActive: 'scan' },
  Collection: { label: 'Colección', icon: 'albums-outline',  iconActive: 'albums' },
};

function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => {
        const cfg = TAB_CONFIG[route.name as keyof TabParamList];
        return {
          headerShown: false,
          tabBarStyle: [
            styles.tabBar,
            // Ocultar tab bar en ScanScreen para cámara a pantalla completa
            route.name === 'Scan' && styles.tabBarHidden,
          ],
          tabBarActiveTintColor:   COLORS.gold,
          tabBarInactiveTintColor: COLORS.textMuted,
          tabBarLabelStyle: styles.tabLabel,
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons
              name={focused ? cfg.iconActive : cfg.icon}
              size={size}
              color={color}
            />
          ),
          tabBarLabel: cfg.label,
        };
      }}
    >
      <Tab.Screen name="Home"       component={HomeScreen} />
      <Tab.Screen name="Scan"       component={ScanScreen} />
      <Tab.Screen name="Collection" component={CollectionScreen} />
    </Tab.Navigator>
  );
}

// ─── Root Stack ───────────────────────────────────────────────────────────────

export function AppNavigator() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Main" component={TabNavigator} />
      <Stack.Screen
        name="Result"
        component={ResultScreen}
        options={{
          presentation:   'formSheet',
          animation:      'slide_from_bottom',
          sheetAllowedDetents: [1.0],
        }}
      />
    </Stack.Navigator>
  );
}

// ─── Estilos ──────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor:  COLORS.surface,
    borderTopColor:   COLORS.border,
    borderTopWidth:   1,
    paddingBottom:    Platform.OS === 'ios' ? 20 : 8,
    paddingTop:       8,
    height:           Platform.OS === 'ios' ? 84 : 64,
  },
  tabBarHidden: {
    display: 'none',
  },
  tabLabel: {
    fontSize:     10,
    letterSpacing: 0.5,
    marginBottom: 2,
  },
});
