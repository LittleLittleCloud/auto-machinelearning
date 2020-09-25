// <copyright file="PipelineBuilderTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using MLNet.Expert;
using Newtonsoft.Json;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.Expert.Tests
{
    public class PipelineBuilderTest : TestBase
    {
        public PipelineBuilderTest(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void PipelineBuilder_should_create_serializable_pipeline()
        {
            var context = new MLContext();
            var columns = new Column[]
            {
                new Column("cat1", ColumnType.Catagorical, ColumnPurpose.CategoricalFeature),
                new Column("numeric1", ColumnType.Numeric, ColumnPurpose.NumericFeature),
                new Column("text1", ColumnType.Numeric, ColumnPurpose.TextFeature),
                new Column("label", ColumnType.Catagorical, ColumnPurpose.Label),
            };

            var pb = new PipelineBuilder(TaskType.BinaryClassification, false, true);
            var pipeline = pb.BuildPipeline(context, columns);

            var pipelineDataContract = pipeline.ToDataContract();
            var json = JsonConvert.SerializeObject(pipelineDataContract, Formatting.Indented);
            Approvals.Verify(json);
            var restoredPipeline = pipelineDataContract.ToPipeline(context);
            var restoredJson = JsonConvert.SerializeObject(restoredPipeline.ToDataContract(), Formatting.Indented);
            restoredJson.Should().Be(json);
        }
    }
}
