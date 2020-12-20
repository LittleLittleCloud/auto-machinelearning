// <copyright file="UtilTests.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML;
using MLNet.Expert.Contract;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.Expert.Tests
{
    public class UtilTests : TestBase
    {
        private MLContext context;

        public UtilTests(ITestOutputHelper output)
            : base(output)
        {
            this.context = new MLContext();
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void SingleEstimatorPipelineContract_to_pipeline_test_0()
        {
            var file = this.GetFileFromTestData("single_estimator_pipeline_contract_1.json");
            var json = File.ReadAllText(file);
            var contract = JsonConvert.DeserializeObject<SingleEstimatorSweepablePipelineDataContract>(json);
            var pipeline = contract.ToPipeline(this.context);
            contract = pipeline.ToDataContract();
            Approvals.Verify(JsonConvert.SerializeObject(contract, Formatting.Indented));
        }

        [Fact]
        public void stringTest()
        {
            var str1 = "new string";
            var str2 = "new string" as object;
            (str1 == str2.ToString()).Should().BeTrue();
        }
    }
}
