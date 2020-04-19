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
    internal class ParameterAttribute : Attribute
    {
        public ParameterAttribute(string name, int min, int max, bool logBase = false, int steps = 100)
        {
            var option = new Int32ValueGenerator.Option()
            {
                Name = name,
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new Int32ValueGenerator(option);
        }

        public ParameterAttribute(string name, float min, float max, bool logBase = false, int steps = 100)
        {
            var option = new FloatValueGenerator.Option()
            {
                Name = name,
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
            };

            this.ValueGenerator = new FloatValueGenerator(option);
        }

        public ParameterAttribute(string name, object[] candidates)
        {
            var option = new DiscreteValueGenerator.Option()
            {
                Name = name,
                Values = candidates,
            };

            this.ValueGenerator = new DiscreteValueGenerator(option);
        }

        public IValueGenerator ValueGenerator { get; }
    }
}
