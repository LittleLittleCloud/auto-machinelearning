// <copyright file="GridSearchSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MLNet.Sweeper
{
    /// <summary>
    /// Grid search sweeper, which performs search on grid points in an ordered way.
    /// </summary>
    public class GridSearchSweeper : SweeperBase
    {
        private int start = 0;
        private long total = 0;
        private int[] basePerPipe;
        private long[] accumulateCountPerBase;

        public override object Clone()
        {
            return new GridSearchSweeper();
        }

        /// <summary>
        /// Propose parameter sets that can be used by <see cref="ISweepable{T}.BuildFromParameterSet(Parameters)"/>.
        /// </summary>
        /// <param name="sweepable">objects that implement <see cref="ISweepable"/>.</param>
        /// <param name="maxSweeps">max sweep iteration. If greater than the total grid points, the number of total grid points will be used instead.</param>
        /// <param name="previousRuns">previous run, which will be used to avoid proposing duplicate parameterset if provided.</param>
        /// <returns><see cref="IEnumerable{ParameterSet}"/>.</returns>
        public override IEnumerable<IDictionary<string, string>> ProposeSweeps(ISweepable sweepable, int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            this.start = 0;
            this.basePerPipe = sweepable.SweepableValueGenerators.Select(x => x.Count).ToArray();
            long accumulateCount = 1;
            this.accumulateCountPerBase = Enumerable.Repeat(0L, sweepable.SweepableValueGenerators.Count()).ToArray();
            for (int i = 0; i != sweepable.SweepableValueGenerators.Count(); ++i)
            {
                this.accumulateCountPerBase[i] = accumulateCount;
                accumulateCount *= this.basePerPipe[i];
            }

            this.total = accumulateCount;

            if (maxSweeps > this.total)
            {
                maxSweeps = (int)this.total;
            }

            return base.ProposeSweeps(sweepable, maxSweeps, previousRuns);
        }

        protected override Parameters CreateParamSet(ISweepable sweepable)
        {
            var indexs = this.GetSelectedIndexForEachPipe();
            this.start += 1;
            return new Parameters(sweepable.SweepableValueGenerators.Select((param, i) => param[indexs[i]]));
        }

        private int[] GetSelectedIndexForEachPipe()
        {
            long current = this.start % this.total;
            var res = Enumerable.Repeat(0, this.basePerPipe.Length).ToArray();
            for (int i = this.basePerPipe.Length - 1; i != -1; --i)
            {
                if (this.basePerPipe[i] == 1)
                {
                    res[i] = 0;
                }
                else
                {
                    res[i] = (int)(current / this.accumulateCountPerBase[i]);
                    current = current % this.accumulateCountPerBase[i];
                }
            }

            return res;
        }
    }
}
