// <copyright file="Utils.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Numpy;
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

        /// <summary>
        /// calc (b-a)*x + a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">B, which must greater than A.</param>
        /// <param name="x">X.</param>
        /// <returns>A*X + B.</returns>
        public static double AXPlusB(double a, double b, double x, bool log)
        {
            if (!log)
            {
                return (b - a) * x + a;
            }
            else
            {
                var lnY = (Math.Log(b) - Math.Log(a)) * x + Math.Log(a);
                return Math.Exp(lnY);
            }
        }

        public static NDarray MultiVariateNormal(NDarray mean, NDarray cov)
        {
            // https://peterroelants.github.io/posts/multivariate-normal-primer/
            // since cov is always [1*1], so calc Cholesky decompostion for cov is simple.
            // mean should be a 1-d array.
            var L = np.linalg.cholesky(cov);

            return L.dot(np.random.standard_normal(mean.shape.Dimensions)) + mean;
        }
    }
}
