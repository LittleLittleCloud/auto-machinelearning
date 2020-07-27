// <copyright file="APITest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.AutoPipeline.Test
{
    public class APITest : TestBase
    {
        public APITest(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_naive_bayes_classifier()
        {
            var context = new MLContext();
            var trainer = context.AutoPipeline().MultiClassification.NaiveBayes("label", "feature");
            Approvals.Verify(trainer.ToCodeGenNodeContract());
        }
    }
}
