// <copyright file="RandomGridSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

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
        private readonly int _nGridPoints;

        // This stores the order of the grid points that are to be generated
        // Only used when the total number of parameter combinations is less than maxGridPoints
        // Every grid point is stored as an int representing the position it would be in a flattened grid
        // In other words, for D dimensions d1,...dn, a point x1,...xn is represented as
        // Sum(i=1..n, xi * Prod(j=i+1..n, dj))
        private readonly int[] _permutation;

        // This is a parallel array to the _permutation array and stores the (already generated) parameter sets
        private readonly ParameterSet[] _cache;

        public RandomGridSweeper(IHostEnvironment env, Options options)
            : base(options, env, "RandomGrid")
        {
            this._nGridPoints = 1;
            foreach (var sweptParameter in this.SweepParameters)
            {
                this._nGridPoints *= sweptParameter.Count;
                if (this._nGridPoints > options.MaxGridPoints)
                {
                    this._nGridPoints = 0;
                }
            }

            if (this._nGridPoints != 0)
            {
                this._permutation = Utils.GetRandomPermutation(this.Host.Rand, this._nGridPoints);
                this._cache = new ParameterSet[this._nGridPoints];
            }
        }

        public RandomGridSweeper(IHostEnvironment env, Options options, IValueGenerator[] sweepParameters)
            : base(options, env, sweepParameters, "RandomGrid")
        {
            this._nGridPoints = 1;
            foreach (var sweptParameter in this.SweepParameters)
            {
                this._nGridPoints *= sweptParameter.Count;
                if (this._nGridPoints > options.MaxGridPoints)
                {
                    this._nGridPoints = 0;
                }
            }

            if (this._nGridPoints != 0)
            {
                this._permutation = Utils.GetRandomPermutation(this.Host.Rand, this._nGridPoints);
                this._cache = new ParameterSet[this._nGridPoints];
            }
        }

        public override ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            if (this._nGridPoints == 0)
            {
                return base.ProposeSweeps(maxSweeps, previousRuns);
            }

            var result = new HashSet<ParameterSet>();
            var prevParamSets = (previousRuns != null)
                ? previousRuns.Select(r => r.ParameterSet).ToList()
                : new List<ParameterSet>();
            int iPerm = (prevParamSets.Count - 1) % this._nGridPoints;
            int tries = 0;
            for (int i = 0; i < maxSweeps; i++)
            {
                for (; ;)
                {
                    iPerm = (iPerm + 1) % this._nGridPoints;
                    if (this._cache[iPerm] == null)
                    {
                        this._cache[iPerm] = this.CreateParamSet(this._permutation[iPerm]);
                    }

                    if (tries++ >= this._nGridPoints)
                    {
                        return result.ToArray();
                    }

                    if (!AlreadyGenerated(this._cache[iPerm], prevParamSets))
                    {
                        break;
                    }
                }

                result.Add(this._cache[iPerm]);
            }

            return result.ToArray();
        }

        protected override ParameterSet CreateParamSet()
        {
            return new ParameterSet(this.SweepParameters.Select(sweepParameter => sweepParameter[this.Host.Rand.Next(sweepParameter.Count)]));
        }

        private ParameterSet CreateParamSet(int combination)
        {
            int div = this._nGridPoints;
            var pset = new List<IParameterValue>();
            foreach (var sweepParameter in this.SweepParameters)
            {
                div /= sweepParameter.Count;
                var pv = sweepParameter[combination / div];
                combination %= div;
                pset.Add(pv);
            }

            return new ParameterSet(pset);
        }

        public sealed class Options : SweeperOptionBase
        {
            public int MaxGridPoints = 1000000;
        }
    }
}
