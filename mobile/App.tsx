import 'react-native-gesture-handler';
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { StatusBar } from 'expo-status-bar';
import { COLORS } from './src/theme';
import { AppNavigator } from './src/navigation';

const NAV_THEME = {
  dark: true,
  colors: {
    primary:      COLORS.gold,
    background:   COLORS.background,
    card:         COLORS.surface,
    text:         COLORS.textPrimary,
    border:       COLORS.border,
    notification: COLORS.primary,
  },
};

export default function App() {
  return (
    <NavigationContainer theme={NAV_THEME}>
      <StatusBar style="light" backgroundColor={COLORS.background} />
      <AppNavigator />
    </NavigationContainer>
  );
}
