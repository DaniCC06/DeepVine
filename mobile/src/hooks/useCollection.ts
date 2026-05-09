import { useState, useEffect, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { TOTAL_VINES } from '../data/vineDatabase';

const STORAGE_KEY = '@deepvine_collection_v1';

export interface CollectionEntry {
  vineId: string;
  discoveredAt: string; // ISO date string
  scansCount: number;
}

interface CollectionState {
  collection: CollectionEntry[];
  total: number;
  loading: boolean;
  isDiscovered: (vineId: string) => boolean;
  addDiscovery: (vineId: string) => Promise<boolean>; // returns true if NEW discovery
  incrementScan: (vineId: string) => Promise<void>;
  resetCollection: () => Promise<void>;
}

export function useCollection(): CollectionState {
  const [collection, setCollection] = useState<CollectionEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCollection();
  }, []);

  async function loadCollection() {
    try {
      const raw = await AsyncStorage.getItem(STORAGE_KEY);
      if (raw) setCollection(JSON.parse(raw) as CollectionEntry[]);
    } catch (e) {
      console.warn('Error cargando colección:', e);
    } finally {
      setLoading(false);
    }
  }

  async function persist(updated: CollectionEntry[]) {
    setCollection(updated);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  }

  const isDiscovered = useCallback(
    (vineId: string) => collection.some((e) => e.vineId === vineId),
    [collection],
  );

  const addDiscovery = useCallback(
    async (vineId: string): Promise<boolean> => {
      const alreadyHave = collection.some((e) => e.vineId === vineId);
      if (alreadyHave) {
        await incrementScan(vineId);
        return false;
      }
      const entry: CollectionEntry = {
        vineId,
        discoveredAt: new Date().toISOString(),
        scansCount: 1,
      };
      await persist([...collection, entry]);
      return true; // nueva variedad
    },
    [collection],
  );

  const incrementScan = useCallback(
    async (vineId: string) => {
      const updated = collection.map((e) =>
        e.vineId === vineId ? { ...e, scansCount: e.scansCount + 1 } : e,
      );
      await persist(updated);
    },
    [collection],
  );

  const resetCollection = useCallback(async () => {
    await persist([]);
  }, []);

  return {
    collection,
    total: TOTAL_VINES,
    loading,
    isDiscovered,
    addDiscovery,
    incrementScan,
    resetCollection,
  };
}
