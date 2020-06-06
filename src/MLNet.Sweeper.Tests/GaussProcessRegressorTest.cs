// <copyright file="GaussProcessRegressorTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using MLNet.Sweeper;
using Numpy;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace MLNet.Sweeper.Tests
{
    public class GaussProcessRegressorTest
    {
        [Fact]
        public void RBF_Test()
        {
            var x1 = np.random.standard_normal(3, 2);
            var x2 = np.random.standard_normal(5, 2);
            var x1_1 = x1["1,:"];
            var x2_1 = x2["1,:"];
            var res = GaussProcessRegressor.RBF(x1, x2, 1, 1);
            res.shape.Dimensions.Should().BeEquivalentTo(new int[] { 3, 5 });
            ((double)res[1, 1]).Should().BeApproximately((double)np.exp(-0.5 * np.dot(np.subtract(x1_1, x2_1), np.subtract(x1_1, x2_1).T)), 0.01);
        }

        [Fact]
        public void GaussProcessRegression_test()
        {
            var X_train = np.arange(0, 100, 1).reshape(-1, 1);
            var y_train = np.sin(X_train);
            var option = new GaussProcessRegressor.Options();
            var GP = new GaussProcessRegressor(option);
            var res = GP.Fit(X_train, y_train).Transform(np.array<double>(new double[] { 10 }).reshape(-1, 1)).Item1.reshape(-1,1);
            ((double)res[0, 0]).Should().BeApproximately(Math.Sin(10), 0.1);
        }
    }
}
