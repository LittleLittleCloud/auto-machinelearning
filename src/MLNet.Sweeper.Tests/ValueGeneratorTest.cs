// <copyright file="ValueGeneratorTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using System;
using Xunit;

namespace MLNet.Sweeper.Tests
{
    public class ValueGeneratorTest
    {
        [Theory]
        [InlineData(-10, 10, 20, false, 0)]
        [InlineData(-10, 10, 15, false, 0)]
        [InlineData(1, 64, 7, true, 8)]
        public void DoubleValueGenerator_should_generate_value_from_normalize(double min, double max, int step, bool logbase, double expect)
        {
            var option = new DoubleValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Name = "double",
                Steps = step,
                LogBase = logbase,
            };

            var generator = new DoubleValueGenerator(option);

            generator.CreateFromNormalized(1.0f).RawValue.As<double>().Should().BeApproximately(max, 0.0001);
            generator.CreateFromNormalized(0f).RawValue.As<double>().Should().BeApproximately(min, 0.0001);
            generator.CreateFromNormalized(0.5f).RawValue.As<double>().Should().BeApproximately(expect, 0.0001);
        }

        [Theory]
        [InlineData(-10, 10, 20, false, 20)]
        [InlineData(-10, 10, 13, false, 13)]
        [InlineData(1, 10, 3, true, 3)]
        public void DoubleValueGenerator_should_work_with_index(double min, double max, int step, bool logBase, int count)
        {
            var option = new DoubleValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = step,
                Name = "double",
                LogBase = logBase,
            };

            var generator = new DoubleValueGenerator(option);

            generator.Count.Should().Be(count + 1);
            generator[0].RawValue.Should().Be(min);
            ((double)generator[count].RawValue)
                .Should()
                .BeApproximately(max, 0.00001);
        }

        [Theory]
        [InlineData(-10f, 10f, 20, false, 0f)]
        [InlineData(-10f, 10f, 15, false, 0f)]
        [InlineData(1f, 64f, 7, true, 8f)]
        public void FloatValueGenerator_should_generate_value_from_normalize(float min, float max, int step, bool logbase, float expect)
        {
            var option = new FloatValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Name = "float",
                Steps = step,
                LogBase = logbase,
            };

            var generator = new FloatValueGenerator(option);

            generator.CreateFromNormalized(1.0f).RawValue.Should().Be(max);
            generator.CreateFromNormalized(0f).RawValue.Should().Be(min);
            generator.CreateFromNormalized(0.5f).RawValue.Should().Be(expect);
        }

        [Theory]
        [InlineData(-10f, 10f, 20, false, 20)]
        [InlineData(-10f, 10f, 13, false, 13)]
        [InlineData(1f, 10f, 3, true, 3)]
        public void FloatValueGenerator_should_work_with_index(float min, float max, int step, bool logBase, int count)
        {
            var option = new FloatValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = step,
                Name = "float",
                LogBase = logBase,
            };

            var generator = new FloatValueGenerator(option);

            generator.Count.Should().Be(count + 1);
            generator[0].RawValue.Should().Be(min);
            ((float)generator[count].RawValue)
                .Should()
                .Be(max);
        }

        [Theory]
        [InlineData(-10, 10, 20, false, 0)]
        [InlineData(-10, 10, 15, false, 0)]
        [InlineData(1, 64, 7, true, 8)]
        public void LongValueGenerator_should_generate_value_from_normalize(long min, long max, int step, bool logbase, long expect)
        {
            var option = new LongValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Name = "long",
                Steps = step,
                LogBase = logbase,
            };

            var generator = new LongValueGenerator(option);

            generator.CreateFromNormalized(1.0).RawValue.Should().Be(max);
            generator.CreateFromNormalized(0).RawValue.Should().Be(min);
            generator.CreateFromNormalized(0.5).RawValue.Should().Be(expect);
        }

        [Theory]
        [InlineData(-10, 10, 20, false, 20)]
        [InlineData(-10, 10, 13, false, 13)]
        [InlineData(1, 10, 3, true, 3)]
        public void LongValueGenerator_should_work_with_index(long min, long max, int step, bool logBase, int count)
        {
            var option = new LongValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = step,
                Name = "long",
                LogBase = logBase,
            };

            var generator = new LongValueGenerator(option);

            generator.Count.Should().Be(count + 1);
            generator[0].RawValue.Should().Be(min);
            ((long)generator[count].RawValue)
                .Should()
                .Be(max);
        }

        [Theory]
        [InlineData(-10, 10, 20, false, 0)]
        [InlineData(-10, 10, 15, false, 0)]
        [InlineData(1, 64, 7, true, 8)]
        public void Int32ValueGenerator_should_generate_value_from_normalize(int min, int max, int step, bool logbase, int expect)
        {
            var option = new Int32ValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Name = "int",
                Steps = step,
                LogBase = logbase,
            };

            var generator = new Int32ValueGenerator(option);

            generator.CreateFromNormalized(1.0).RawValue.Should().Be(max);
            generator.CreateFromNormalized(0).RawValue.Should().Be(min);
            generator.CreateFromNormalized(0.5).RawValue.Should().Be(expect);
        }

        [Theory]
        [InlineData(-10, 10, 20, false, 20)]
        [InlineData(-10, 10, 13, false, 13)]
        [InlineData(1, 10, 3, true, 3)]
        public void Int32ValueGenerator_should_work_with_index(int min, int max, int step, bool logBase, int count)
        {
            var option = new Int32ValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = step,
                Name = "int",
                LogBase = logBase,
            };

            var generator = new Int32ValueGenerator(option);

            generator.Count.Should().Be(count + 1);
            generator[0].RawValue.Should().Be(min);
            ((int)generator[count].RawValue)
                .Should()
                .Be(max);
        }

        [Theory]
        [InlineData("str1", "str2", 3, 4)]
        public void DiscreteValueGenerator_should_generate_value_from_normalize(object a, object b, object c, object d)
        {
            var objects = new object[] { a, b, c, d };
            var option = new DiscreteValueGenerator<object>.Option<object>()
            {
                Name = "discrete",
                Values = objects,
            };

            var generator = new DiscreteValueGenerator<object>(option);

            objects.Should().Contain(generator.CreateFromNormalized(0.5).RawValue);
            generator.Count.Should().Be(4);
        }

        [Fact]
        public void DiscreteValueGenerator_should_return_one_hot_encode()
        {
            var objects = new object[] { "a", 2, "c", 4 };
            var option = new DiscreteValueGenerator<object>.Option<object>()
            {
                Name = "discrete",
                Values = objects,
            };

            var generator = new DiscreteValueGenerator<object>(option);

            generator.OneHotEncodeValue(new ObjectParameterValue<object>("val", objects[0])).Should().BeEquivalentTo(new int[] { 1, 0, 0, 0 });
        }
    }
}
