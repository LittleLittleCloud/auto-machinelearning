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

        public static NDarray NormPDF(NDarray X)
        {
            return (1 / np.sqrt((NDarray)2 * Math.PI)) * np.exp(-0.5 * X * X);
        }

        public static NDarray NormCDF(NDarray X)
        {
            return 0.5 * (1 + ERF(X / np.sqrt((NDarray)2)));
        }

        public static double NormCDF(double X)
        {
            return 0.5 * (1 + ERF(X / Math.Sqrt(2)));
        }

        public static double Normal()
        {
            var seed = new Random();
            var u1 = seed.NextDouble();
            var u2 = seed.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        public static NDarray ERF(NDarray X)
        {
            // https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
            var sign = np.ones(X.shape);
            sign[X < 0] = (NDarray)(-1);

            var a1 = (NDarray)0.254829592;
            var a2 = (NDarray)(-0.284496736);
            var a3 = (NDarray)1.421413741;
            var a4 = (NDarray)(-1.453152027);
            var a5 = (NDarray)1.061405429;
            var p = (NDarray)0.3275911;

            var t = 1.0 / (1.0 + p * X);
            var y = (NDarray)1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-X * X);
            return sign * y;
        }

        public static double ERF(double X)
        {
            // https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
            var sign = X > 0 ? 1.0 : -1.0; 

            var a1 = 0.254829592;
            var a2 = -0.284496736;
            var a3 = 1.421413741;
            var a4 = -1.453152027;
            var a5 = 1.061405429;
            var p = 0.3275911;

            var t = 1.0 / (1.0 + p * X);
            var y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-X * X);
            return sign * y;
        }

        public static ObjectParameterValue<T> CreateObjectParameterValue<T>(string name, T value, double[] onehot, string id = null)
        {
            return new ObjectParameterValue<T>(name, value, onehot, id);
        }
    }
}
