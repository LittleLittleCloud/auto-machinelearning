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

        public RandomGridSweeper()
            : base()
        {
            this._rand = new Random();
            this._option = new Option();
        }

        public override object Clone()
        {
            return new RandomGridSweeper(this._option);
        }

        protected override Parameters CreateParamSet(ISweepable sweepable)
        {
            return new Parameters(sweepable.SweepableValueGenerators.Select(sweepParameter => sweepParameter[this._rand.Next(sweepParameter.Count)]));
        }

        public sealed class Option : SweeperOptionBase
        {
        }
    }
}
