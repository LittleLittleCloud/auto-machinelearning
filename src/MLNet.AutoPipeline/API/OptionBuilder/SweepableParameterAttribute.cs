// <copyright file="ParameterAttribute.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public class SweepableParameterAttribute : Attribute
    {
        public SweepableParameterAttribute(int min, int max, bool logBase = false, int steps = 100)
        {
            Contract.Assert(max > min);
            Contract.Assert(steps > 0);
            Contract.Assert(!logBase || (logBase && min > 0));
            var option = new Int32ValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new Int32ValueGenerator(option);
        }

        public SweepableParameterAttribute(long min, long max, bool logBase = false, int steps = 100)
        {
            Contract.Assert(max > min);
            Contract.Assert(steps > 0);
            Contract.Assert(!logBase || (logBase && min > 0));
            var option = new LongValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new LongValueGenerator(option);
        }

        public SweepableParameterAttribute(float min, float max, bool logBase = false, int steps = 100)
        {
            Contract.Assert(max > min);
            Contract.Assert(steps > 0);
            Contract.Assert(!logBase || (logBase && min > 0));
            var option = new FloatValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new FloatValueGenerator(option);
        }

        public SweepableParameterAttribute(object[] candidates)
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
