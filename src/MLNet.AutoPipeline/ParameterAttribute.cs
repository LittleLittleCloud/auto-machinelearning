// <copyright file="ParameterAttribute.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public class ParameterAttribute : Attribute
    {
        public ParameterAttribute(int min, int max, bool logBase = false, int steps = 100)
        {
            var option = new Int32ValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new Int32ValueGenerator(option);
        }

        public ParameterAttribute(long min, long max, bool logBase = false, int steps = 100)
        {
            var option = new LongValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new LongValueGenerator(option);
        }

        public ParameterAttribute(float min, float max, bool logBase = false, int steps = 100)
        {
            var option = new FloatValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new FloatValueGenerator(option);
        }

        public ParameterAttribute(object[] candidates)
        {
            var option = new DiscreteValueGenerator.Option()
            {
                Values = candidates,
            };

            this.ValueGenerator = new DiscreteValueGenerator(option);
        }

        public IValueGenerator ValueGenerator { get; }
    }
}
