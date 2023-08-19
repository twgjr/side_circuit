import 'package:flutter/material.dart';

ColorScheme kColorScheme = ColorScheme.fromSeed(
  seedColor: Colors.blue,
  brightness: Brightness.light,
);

ColorScheme kDarkColorScheme = ColorScheme.fromSeed(
  seedColor: Colors.blue,
  brightness: Brightness.dark,
);

ThemeData lightTheme = ThemeData().copyWith(
  useMaterial3: true,
  colorScheme: kColorScheme,
  appBarTheme: AppBarTheme().copyWith(
    backgroundColor: kColorScheme.onPrimaryContainer,
    foregroundColor: kColorScheme.primaryContainer,
  ),
  cardTheme: const CardTheme().copyWith(
    color: kColorScheme.secondaryContainer,
    margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
  ),
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      backgroundColor: kColorScheme.primaryContainer,
      foregroundColor: kColorScheme.onPrimaryContainer,
    ),
  ),
  textTheme: ThemeData().textTheme.copyWith(
        titleLarge: TextStyle(
          fontWeight: FontWeight.normal,
          fontSize: 14,
          color: kColorScheme.onSecondaryContainer,
        ),
      ),
);

ThemeData darkTheme = ThemeData.dark().copyWith(
  useMaterial3: true,
  colorScheme: kDarkColorScheme,
  appBarTheme: const AppBarTheme().copyWith(
    backgroundColor: kDarkColorScheme.secondary,
    foregroundColor: kDarkColorScheme.onSecondaryContainer,
  ),
  cardTheme: const CardTheme().copyWith(
    color: kDarkColorScheme.secondaryContainer,
    margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
  ),
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      backgroundColor: kDarkColorScheme.primaryContainer,
      foregroundColor: kDarkColorScheme.onPrimaryContainer,
    ),
  ),
  textTheme: ThemeData().textTheme.copyWith(
        titleLarge: TextStyle(
          fontWeight: FontWeight.normal,
          fontSize: 14,
          color: kDarkColorScheme.onSecondaryContainer,
        ),
      ),
);
