// <copyright file="UniformRandomSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;

namespace MLNet.Sweeper
{
    /// <summary>
    /// Random sweeper, it generates random values for each of the parameters.
    /// </summary>
    public sealed class UniformRandomSweeper : SweeperBase
    {
        private readonly Random _rand;
        private readonly Option _option;

        public UniformRandomSweeper(Option options)
            : base(options, "UniformRandom")
        {
            this._option = options;
            this._rand = new Random();
        }

        protected override ParameterSet CreateParamSet()
        {
            return new ParameterSet(this.SweepableParamaters.Select(sweepParameter => sweepParameter.CreateFromNormalized(this._rand.NextDouble())));
        }

        public class Option : SweeperOptionBase
        {
        }
    }
}
