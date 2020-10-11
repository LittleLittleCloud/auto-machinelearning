// <copyright file="OptionBuilderTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML;
using MLNet.Sweeper;
using System.Collections.Generic;
using Xunit;

namespace MLNet.AutoPipeline.Test
{
    public class OptionBuilderTest
    {
        [Fact]
        public void OptionBuilder_should_create_default_option()
        {
            var builder = new TestOptionBuilderWithSweepableAttributeOnly();
            var option = builder.CreateDefaultOption();
            option.LongOption.Should().Equals(10);
            option.FloatOption.Should().Equals(1f);
            option.StringOption.Should().Equals("str");
        }

        [Fact]
        public void OptionBuilder_should_build_option_from_parameter_set()
        {
            var builder = new TestOptionBuilderWithSweepableAttributeOnly();
            var input = new Dictionary<string, string>()
            {
                { "LongOption", "2" },
                { "FloatOption", "2" },
                { "StringOption", "str2" },
            };

            var option = builder.BuildFromParameters(input);
            option.LongOption.Should().Equals(2);
            option.FloatOption.Should().Equals(2f);
            option.StringOption.Should().Equals("str2");
        }

        [Fact]
        public void OptionBuilder_should_work_with_random_sweeper()
        {
            var context = new MLContext();
            var builder = new TestOptionBuilderWithSweepableAttributeOnly();
            var maximum = 10;
            var sweeperOption = new UniformRandomSweeper.Option();

            var randomSweeper = new UniformRandomSweeper(sweeperOption);

            foreach (var sweeperOutput in randomSweeper.ProposeSweeps(builder, maximum))
            {
                maximum -= 1;
                var option = builder.BuildFromParameters(sweeperOutput);
                option.LongOption
                      .Should()
                      .BeLessOrEqualTo(100)
                      .And
                      .BeGreaterOrEqualTo(0);

                option.FloatOption
                      .Should()
                      .BeLessOrEqualTo(100f)
                      .And
                      .BeGreaterOrEqualTo(0f);

                option.StringOption
                      .Should()
                      .BeOneOf(new string[] { "str1", "str2", "str3", "str4" });

                maximum.Should().BeGreaterThan(-2);
            }
        }

        [Fact]
        public void OptionBuilder_should_build_option_using_field_with_parameter_attribute()
        {
            var optionBuilder = new TestOptionBuilderWithParameterAttributeOnly();
            var option1 = optionBuilder.CreateDefaultOption();
            option1.FloatOption.Should().Be(100f);
            option1.LongOption.Should().Be(100L);
            option1.StringOption.Should().Be(string.Empty);
        }

        [Fact]
        public void OptionBuilder_should_build_option_using_field_with_sweepable_parameter_attribute()
        {
            var optionBuilder = new TestOptionBuilderWithSweepableAttributeOnly();
            var option1 = optionBuilder.CreateDefaultOption();
            option1.FloatOption.Should().Be(0f);
            option1.LongOption.Should().Be(0);
            option1.StringOption.Should().Be("str1");

            var input = new Dictionary<string, string>()
            {
                { "LongOption", "2" },
                { "FloatOption", "2" },
                { "StringOption", "str2" },
            };

            var option2 = optionBuilder.BuildFromParameters(input);

            option2.LongOption.Should().Equals(2);
            option2.FloatOption.Should().Equals(2f);
            option2.StringOption.Should().Equals("str2");
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void OptionBuilder_should_print_in_a_human_readable_format()
        {
            var optionBuilder = new TestOptionBuilderWithSweepableAttributeOnly();
            Approvals.Verify(optionBuilder);
        }

        [Fact]
        public void SweepableTestOption_should_use_matching_field_to_create_default_option()
        {
            var option = new SweepableTestOption();
            option.LongOption = 100f;
            option.FloatOption = 100f;
            option.StringOption = "123";

            // long option should still be 1L, because the type of LongOption in SweepableTestOption
            // is float, which doesn't match with TestOption.
            option.CreateDefaultOption().LongOption.Should().Be(1);

            // float option should still be 100f, because the type of FloatOption in SweepableTestOption
            // is float, which matches with TestOption.
            option.CreateDefaultOption().FloatOption.Should().Be(100f);

            // string option should still be empty, even the type and field name match with stringoption, the name
            // in Parameter attribute doesn't match.
            option.CreateDefaultOption().StringOption.Should().Be(string.Empty);
        }

        private class TestOption
        {
            public long LongOption = 1;

            public float FloatOption = 1f;

            public string StringOption = string.Empty;
        }

        private class TestOptionBuilderWithParameterAttributeOnly : SweepableOption<TestOption>
        {
            [Parameter]
            public Parameter<long> LongOption = ParameterFactory.CreateDiscreteParameter(100L);

            [Parameter(nameof(TestOption.FloatOption))]
            public Parameter<float> Float_Option = ParameterFactory.CreateDiscreteParameter(100f);
        }

        private class TestOptionBuilderWithSweepableAttributeOnly : SweepableOption<TestOption>
        {
            [Parameter]
            public Parameter<long> LongOption = ParameterFactory.CreateLongParameter(0, 100);

            [Parameter(nameof(TestOption.FloatOption))]
            public Parameter<float> Float_Option = ParameterFactory.CreateFloatParameter(0f, 100f);

            [Parameter]
            public Parameter<string> StringOption = ParameterFactory.CreateDiscreteParameter("str1", "str2", "str3", "str4");
        }

        private class SweepableTestOption : SweepableOption<TestOption>
        {
            [Parameter]
            public float LongOption = 1f;

            [Parameter]
            public float FloatOption = 1f;

            [Parameter(nameof(FloatOption))]
            public string StringOption;
        }
    }
}
