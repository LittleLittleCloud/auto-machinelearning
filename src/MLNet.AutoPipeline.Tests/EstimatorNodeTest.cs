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
    public class EstimatorNodeTest
    {
        private ITestOutputHelper _output;

        public EstimatorNodeTest(ITestOutputHelper output)
        {
            this._output = output;
        }

        [Fact]
        public void EstimatorSingleNode_summary_should_work()
        {
            var singleNode = new EstimatorSingleNode(new MockEstimatorBuilder("MockEstiamtor"));
            singleNode.Summary().Should().Be("SingleNode(MockEstiamtor)");
            singleNode.NodeType.Should().Be(EstimatorNodeType.Node);
        }

        [Fact]
        public void EstimatorNodeGroup_summary_should_work()
        {
            var autoEstimatorBuilder = new MockEstimatorBuilder("mockEstimator");
            var estimatorWrapper = new UnsweepableNode<ITransformer>(new MockTransformer());

            var builders = new IEstimatorNode[]
            {
                new EstimatorSingleNode(autoEstimatorBuilder),
                new EstimatorSingleNode(estimatorWrapper),
            };

            var mixNode = new EstimatorNodeGroup(builders);

            var autoEstimatorChain = new EstimatorNodeChain()
                            .Append(new EstimatorSingleNode(autoEstimatorBuilder))
                            .Append(mixNode);

            mixNode.Summary().Should().Be("NodeGroup(SingleNode(mockEstimator), SingleNode(MockTransformer))");
            mixNode.NodeType.Should().Be(EstimatorNodeType.NodeGroup);
        }

        [Fact]
        public void EstimatorNodeChain_summary_should_work()
        {
            var autoEstimatorBuilder = new MockEstimatorBuilder("mockEstimator");
            var estimatorWrapper = new UnsweepableNode<ITransformer>(new MockTransformer());

            var builders = new IEstimatorNode[]
            {
                new EstimatorSingleNode(autoEstimatorBuilder),
                new EstimatorSingleNode(estimatorWrapper),
            };

            var mixNode = new EstimatorNodeGroup(builders);

            var autoEstimatorChain = new EstimatorNodeChain()
                            .Append(new EstimatorSingleNode(autoEstimatorBuilder))
                            .Append(mixNode);
            autoEstimatorChain.Summary().Should().Be("NodeChain(SingleNode(mockEstimator)=>NodeGroup(SingleNode(mockEstimator), SingleNode(MockTransformer)))");
            autoEstimatorChain.NodeType.Should().Be(EstimatorNodeType.NodeChain);
        }

        [Fact]
        public void EstimatorSingleNode_should_build_sweepable_pipeline()
        {
            var singleNode = new EstimatorSingleNode(new MockEstimatorBuilder("MockEstiamtor"));
            singleNode.BuildSweepablePipelines().Count().Should().Be(1);
            singleNode.BuildSweepablePipelines().First().Summary().Should().Be("SweepablePipeline(MockEstiamtor)");
        }

        [Fact]
        public void EstimatorNodeGroup_should_build_sweepable_pipeline()
        {
            var singleNode1 = new EstimatorSingleNode(new MockEstimatorBuilder("MockEstimator"));
            var singleNode2 = new EstimatorSingleNode(new UnsweepableNode<ITransformer>(new MockTransformer()));
            var singleNode3 = new EstimatorSingleNode(new MockEstimatorBuilder("MockEstimator1"));
            var nodeGroup1 = new EstimatorNodeGroup().Append(singleNode1).Append(singleNode2).Append(singleNode3);
            var nodechain = new EstimatorNodeChain().Append(singleNode1).Append(singleNode2).Append(nodeGroup1);

            var nodeGroup = new EstimatorNodeGroup().Append(singleNode1).Append(singleNode2).Append(nodechain);
            var estimatorChains = nodeGroup.BuildSweepablePipelines().ToArray();
            estimatorChains.Length.Should().Be(5);
            estimatorChains[0].Summary().Should().Be("SweepablePipeline(MockEstimator)");
            estimatorChains[1].Summary().Should().Be("SweepablePipeline(MockTransformer)");
            estimatorChains[2].Summary().Should().Be("SweepablePipeline(MockEstimator=>MockTransformer=>MockEstimator)");
            estimatorChains[3].Summary().Should().Be("SweepablePipeline(MockEstimator=>MockTransformer=>MockTransformer)");
            estimatorChains[4].Summary().Should().Be("SweepablePipeline(MockEstimator=>MockTransformer=>MockEstimator1)");
        }
    }
}
