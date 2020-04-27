// <copyright file="GaussProcessRegressor.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using Numpy;

namespace MLNet.Sweeper.GP
{
    public class GaussProcessRegressor
    {
        private NDarray xTrain;
        private NDarray yTrain;

        public Options Option { get; private set; }

        public GaussProcessRegressor(Options options)
        {
            this.Option = options;
        }

        public GaussProcessRegressor Fit(NDarray X, NDarray y)
        {
            this.xTrain = X;
            this.yTrain = y;
            return this;
        }

        public (NDarray, NDarray, NDarray) Transform(NDarray X)
        {
            var k = GaussProcessRegressor.RBF(this.xTrain, this.xTrain, this.Option.l, this.Option.sigma) + this.Option.noise * this.Option.noise * np.eye(this.xTrain.len);
            var k_s = GaussProcessRegressor.RBF(this.xTrain, X, this.Option.l, this.Option.sigma);
            var k_ss = GaussProcessRegressor.RBF(X, X, this.Option.l, this.Option.sigma) + 1e-8 * np.eye(X.len);
            var k_inv = np.linalg.inv(k);

            var mean_s = k_s.T.dot(k_inv).dot(this.yTrain);
            var cov_s = k_ss - k_s.T.dot(k_inv).dot(k_s);
            var res = np.random.multivariate_normal(mean_s.ravel(), cov_s);
            return (res, mean_s, cov_s);
        }

        public static NDarray RBF(NDarray x1, NDarray x2, double l = 1, double sigma = 1)
        {
            var sqdist = np.sum(np.power(x1, (NDarray)2), new int[] { 1 }).reshape(-1, 1) + np.sum(np.power(x2, (NDarray)2), new int[] { 1 }) - 2 * np.dot(x1, x2.T);
            return sigma*sigma * np.exp(-0.5 / l * l * sqdist);
        }

        public class Options
        {
            public double l { get; set; } = 1;

            public double sigma { get; set; } = 1;

            public double noise { get; set; } = 1e-5;
        }
    }
}
