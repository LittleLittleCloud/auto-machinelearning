// <copyright file="NumericFeatureExpertTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using Microsoft.ML;
using Xunit;

namespace MLNet.Expert.Tests
{
    public class NumericFeatureExpertTest
    {
        [Fact]
        public void NumericFeatureExpert_should_propose_NormalizeMeanVariance_and_NormalizeMinMax()
        {
            var context = new MLContext();
            var option = new NumericFeatureExpert.Option();

            var expert = new NumericFeatureExpert(context, option);
            expert.Propose("test").ToString()
                  .Should().Contain("SingleNode(NormalizeMeanVariance)")
                  .And
                  .Contain("SingleNode(NormalizeMinMax)");
        }
    }
}
