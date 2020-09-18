// <copyright file="ExpertTests.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.Expert.Tests
{
    public class ExpertTests : TestBase
    {
        public ExpertTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        private class Iris
        {
            [LoadColumn(0)]
            public float sepal_length;

            [LoadColumn(1)]
            public float sepal_width;

            [LoadColumn(2)]
            public float petal_length;

            [LoadColumn(3)]
            public float petal_width;

            [LoadColumn(4)]
            public string species;
        }

        private class Reporter : IProgress<IterationInfo>
        {
            private ITestOutputHelper output;

            public Reporter(ITestOutputHelper output)
            {
                this.output = output;
            }

            public void Report(IterationInfo value)
            {
                this.output.WriteLine(value.Parameters?.ToString() ?? string.Empty);
                this.output.WriteLine($"evaluate score: {value.EvaluateScore}");
                this.output.WriteLine($"training time: {value.TrainingTime}");
            }
        }
    }
}
