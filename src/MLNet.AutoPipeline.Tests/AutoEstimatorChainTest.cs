// <copyright file="AutoEstimatorChainTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using Microsoft.ML;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.Trainers.MatrixFactorizationTrainer;

namespace MLNet.AutoPipeline.Test
{
    public class AutoEstimatorChainTest
    {
        private ITestOutputHelper _output;

        public AutoEstimatorChainTest(ITestOutputHelper output)
        {
            this._output = output;
        }

        [Fact]
        public void AutoEstimatorSingleNode_summary_should_work()
        {
            var singleNode = new AutoEstimatorSingleNode(new MockEstimatorBuilder("MockEstiamtor"));
            singleNode.Summary().Should().Be("SingleNode(MockEstiamtor)");
            singleNode.NodeType.Should().Be(AutoEstimatorNodeType.Node);
        }

        [Fact]
        public void AutoEstimatorNodeGroup_summary_should_work()
        {
            var autoEstimatorBuilder = new MockEstimatorBuilder("mockEstimator");
            var estimatorWrapper = new EstimatorWrapper<ITransformer>(new MockTransformer());

            var builders = new IAutoEstimatorNode[]
            {
                new AutoEstimatorSingleNode(autoEstimatorBuilder),
                new AutoEstimatorSingleNode(estimatorWrapper),
            };

            var mixNode = new AutoEstimatorNodeGroup(builders);

            var autoEstimatorChain = new AutoEstimatorNodeChain()
                            .Append(new AutoEstimatorSingleNode(autoEstimatorBuilder))
                            .Append(mixNode);

            mixNode.Summary().Should().Be("NodeGroup(SingleNode(mockEstimator), SingleNode(ITransformer))");
            mixNode.NodeType.Should().Be(AutoEstimatorNodeType.NodeGroup);
        }

        [Fact]
        public void AutoEstimatorSingleNode_should_build_single_estimator_chain()
        {
            var singleNode = new AutoEstimatorSingleNode(new MockEstimatorBuilder("MockEstiamtor"));
            singleNode.BuildEstimatorChains().Count().Should().Be(1);
            singleNode.BuildEstimatorChains().First().Summary().Should().Be("SingleNodeChain(MockEstiamtor)");
        }

        [Fact]
        public void AutoEstimatorNodeGroup_should_build_estimator_chain()
        {
            var singleNode1 = new AutoEstimatorSingleNode(new MockEstimatorBuilder("MockEstimator"));
            var singleNode2 = new AutoEstimatorSingleNode(new EstimatorWrapper<ITransformer>(new MockTransformer()));
            var singleNode3 = new AutoEstimatorSingleNode(new MockEstimatorBuilder("MockEstimator1"));
            var nodeGroup1 = new AutoEstimatorNodeGroup().Append(singleNode1).Append(singleNode2).Append(singleNode3);
            var nodechain = new AutoEstimatorNodeChain().Append(singleNode1).Append(singleNode2).Append(nodeGroup1);

            var nodeGroup = new AutoEstimatorNodeGroup().Append(singleNode1).Append(singleNode2).Append(nodechain);
            var estimatorChains = nodeGroup.BuildEstimatorChains().ToArray();
            estimatorChains.Length.Should().Be(5);
            estimatorChains[0].Summary().Should().Be("SingleNodeChain(MockEstimator)");
            estimatorChains[1].Summary().Should().Be("SingleNodeChain(ITransformer)");
            estimatorChains[2].Summary().Should().Be("SingleNodeChain(MockEstimator=>ITransformer=>MockEstimator)");
            estimatorChains[3].Summary().Should().Be("SingleNodeChain(MockEstimator=>ITransformer=>ITransformer)");
            estimatorChains[4].Summary().Should().Be("SingleNodeChain(MockEstimator=>ITransformer=>MockEstimator1)");
        }
    }
}
