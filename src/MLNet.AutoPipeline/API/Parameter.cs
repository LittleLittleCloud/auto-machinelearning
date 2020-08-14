// <copyright file="Parameter.cs" company="BigMiao">
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
    /// <summary>
    /// Create a parameter that can be used in <see cref="SweepableOption{TOption}"/>.
    /// </summary>
    /// <typeparam name="T">type of Parameter.</typeparam>
    public class IParameter<T> : IParameter
    {
        internal IParameter(int min, int max, bool logBase = false, int steps = 100)
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

        internal IParameter(long min, long max, bool logBase = false, int steps = 100)
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

        internal IParameter(float min, float max, bool logBase = false, int steps = 100)
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

        internal IParameter(double min, double max, bool logBase = false, int steps = 100)
        {
            Contract.Assert(max > min);
            Contract.Assert(steps > 0);
            Contract.Assert(!logBase || (logBase && min > 0));
            var option = new DoubleValueGenerator.Option()
            {
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new DoubleValueGenerator(option);
        }

        internal IParameter(T[] candidates)
        {
            var option = new DiscreteValueGenerator.Option()
            {
                Values = candidates.Select(x => (object)x).ToArray(),
            };

            this.ValueGenerator = new DiscreteValueGenerator(option);
        }

        internal IParameter(T value)
        {
            this.ValueGenerator = new SingleValueGenerator<T>(null, value);
        }
    }

    /// <summary>
    /// </summary>
    public interface IParameter
    {
        IValueGenerator ValueGenerator { public get; }
    }
}
