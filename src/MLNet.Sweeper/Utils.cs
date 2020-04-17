// <copyright file="Utils.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;

namespace MLNet.Sweeper
{
    public static class Utils
    {
        public static int[] GetIdentityPermutation(int size)
        {
            var res = new int[size];
            for (int i = 0; i < size; i++)
            {
                res[i] = i;
            }

            return res;
        }

        public static int[] GetRandomPermutation(Random rand, int size)
        {

            var res = GetIdentityPermutation(size);
            Shuffle<int>(rand, res);
            return res;
        }

        public static void Shuffle<T>(Random rand, Span<T> rgv)
        {
            for (int iv = 0; iv < rgv.Length; iv++)
            {
                Swap(ref rgv[iv], ref rgv[iv + rand.Next(rgv.Length - iv)]);
            }
        }

        public static void Swap<T>(ref T a, ref T b)
        {
            T temp = a;
            a = b;
            b = temp;
        }
    }
}
