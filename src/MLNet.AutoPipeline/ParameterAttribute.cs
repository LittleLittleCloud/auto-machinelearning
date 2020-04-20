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
        public ParameterAttribute(string name, int min, int max, bool logBase = false, int steps = 100, string groupID = null)
        {
            var option = new Int32ValueGenerator.Option()
            {
                Name = name,
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
                GroupID = groupID,
            };

            this.GroupID = groupID;

            this.ValueGenerator = new Int32ValueGenerator(option);
        }

        public ParameterAttribute(string name, float min, float max, bool logBase = false, int steps = 100, string groupID = null)
        {
            var option = new FloatValueGenerator.Option()
            {
                Name = name,
                Min = min,
                Max = max,
                Steps = steps,
                LogBase = logBase,
                GroupID = groupID,
            };

            this.GroupID = groupID;

            this.ValueGenerator = new FloatValueGenerator(option);
        }

        public ParameterAttribute(string name, object[] candidates, string groupID = null)
        {
            var option = new DiscreteValueGenerator.Option()
            {
                Name = name,
                Values = candidates,
                GroupID = groupID,
            };

            this.GroupID = groupID;
            this.ValueGenerator = new DiscreteValueGenerator(option);
        }

        public string GroupID { get; private set; }

        public IValueGenerator ValueGenerator { get; }
    }
}
