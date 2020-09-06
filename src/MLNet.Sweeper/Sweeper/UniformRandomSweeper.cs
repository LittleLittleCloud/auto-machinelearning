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

        public override object Clone()
        {
            return new UniformRandomSweeper(this._option);
        }

        protected override Parameters CreateParamSet(ISweepable sweepable)
        {
            return new Parameters(sweepable.SweepableValueGenerators.Select(sweepParameter => sweepParameter.CreateFromNormalized(this._rand.NextDouble())));
        }

        public class Option : SweeperOptionBase
        {
        }
    }
}
