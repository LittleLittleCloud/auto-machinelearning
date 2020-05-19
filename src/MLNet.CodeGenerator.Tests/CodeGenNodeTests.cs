// <copyright file="CodeGenNodeTests.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using System;
using Xunit;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;

namespace MLNet.CodeGenerator.Tests
{
    public class CodeGenNodeTests
    {
        [Theory]
        [InlineData(123, "123")] // Int32
        [InlineData("123", "\"123\"")] // string
        [InlineData(123.45, "123.45F")] // double
        [InlineData(123.45f, "123.45F")] // float
        [InlineData(ParamaterValueType.Enum, "MLNet.CodeGenerator.ParamaterValueType.Enum")] // enum
        [InlineData(new string[] { "123", "456", "789" }, "new string[]{\"123\",\"456\",\"789\"}")] // string array
        [InlineData(new int[] { 123, 456, 789 }, "new int[]{123,456,789}")] // int array
        [InlineData(new float[] { 123f, 456f, 7e-10f }, "new float[]{123F,456F,7E-10F}")] // int array
        [InlineData(new double[] { 123, 456, 7e-10 }, "new float[]{123F,456F,7E-10F}")] // int array
        public void ParamaterValue_should_generate_code_from<T>(T value, string expected)
        {
            var pv = ParamaterValue.Create(value);
            pv.GeneratorCode().Should().Be(expected);
        }

        [Fact]
        public void Paramater_should_generate_code()
        {
            var pv = ParamaterValue.Create(123);
            var paramater = new Paramater("param", pv);

            paramater.GeneratorCode().Should().Be("param=123");
        }

        [Fact]
        public void ParamaterList_should_generate_code()
        {
            var paramList = this.GetParamaterList();

            paramList.GeneratorCode().Should().Be("123=123,456=456,789=789");
        }

        [Fact]
        public void Estimator_should_generate_code()
        {
            this.GetTrainerEstimator().GeneratorCode().Should().Be("context.prefix.trainer(123=123,456=456,789=789)");
            this.GetTransformerEstimator().GeneratorCode().Should().Be("context.prefix.transformer(123=123,456=456,789=789)");
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void EstimatorChain_should_generate_code()
        {
            var trainer = this.GetTrainerEstimator();
            var transformer = this.GetTransformerEstimator();

            var estimatorChain = new EstimatorChain()
            {
                trainer,
                transformer,
            };

            Approvals.Verify(estimatorChain.GeneratorCode());
        }

        private ParamaterList GetParamaterList()
        {
            var paramater1 = new Paramater("123", ParamaterValue.Create(123));
            var paramater2 = new Paramater("456", ParamaterValue.Create(456));
            var paramater3 = new Paramater("789", ParamaterValue.Create(789));

            var paramList = new ParamaterList()
            {
                paramater1,
                paramater2,
                paramater3,
            };

            return paramList;
        }

        private Estimator GetTrainerEstimator()
        {
            var paramList = this.GetParamaterList();
            var estimator = new Estimator("trainer", "prefix", EstimatorType.Trainer, paramList);
            return estimator;
        }

        private Estimator GetTransformerEstimator()
        {
            var paramList = this.GetParamaterList();
            var estimator = new Estimator("transformer", "prefix", EstimatorType.Transformer, paramList);
            return estimator;
        }

    }
}
