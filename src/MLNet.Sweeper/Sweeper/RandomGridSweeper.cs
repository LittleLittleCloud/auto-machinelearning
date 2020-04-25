// <copyright file="RandomGridSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;

namespace MLNet.Sweeper
{
    /// <summary>
    /// Random grid sweeper, it generates random points from the grid.
    /// </summary>
    public sealed class RandomGridSweeper : SweeperBase
    {
        private readonly Random _rand;
        private readonly Option _option;

        public RandomGridSweeper(Option options)
            : base(options, "RandomGrid")
        {
            this._rand = new Random();
            this._option = options;
        }

        protected override ParameterSet CreateParamSet()
        {
            return new ParameterSet(this.SweepableParamaters.Select(sweepParameter => sweepParameter[this._rand.Next(sweepParameter.Count)]));
        }

        public sealed class Option : SweeperOptionBase
        {
        }
    }
}
