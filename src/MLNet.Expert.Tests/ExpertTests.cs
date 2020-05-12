// <copyright file="ExpertTests.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using Microsoft.ML;
using Xunit;

namespace MLNet.Expert.Tests
{
    public class ExpertTests
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

        [Fact]
        public void ClassificationExpert_should_propose_naiveBayes()
        {
            var expert = this.GetClassificationExpert();
            expert.Propose("label", "feature").ToString()
                  .Should().Contain("SingleNode(NaiveBayesMulticlassTrainer)");
        }

        [Fact]
        public void ClassificationExpert_should_propose_lbfgsMaximumEntropy()
        {
            var expert = this.GetClassificationExpert();
            expert.Propose("label", "feature").ToString()
                  .Should().Contain("SingleNode(LbfgsMaximumEntropy)");
        }

        [Fact]
        public void ClassificationExpert_should_propose_lightGBM()
        {
            var expert = this.GetClassificationExpert();
            expert.Propose("label", "feature").ToString()
                  .Should().Contain("SingleNode(LightGBM)");
        }

        private ClassificationExpert GetClassificationExpert()
        {
            var context = new MLContext();
            var expert = new ClassificationExpert(context, new ClassificationExpert.Option());
            return expert;
        }
    }
}
