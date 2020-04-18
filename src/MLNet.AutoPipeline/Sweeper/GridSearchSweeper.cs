// <copyright file="GridSearchSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System.Collections;
using System.Collections.Generic;
using Microsoft.ML;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    internal class GridSearchSweeper : ISweeper
    {
        private readonly RandomGridSweeper _gridSweeper;

        private ParameterSet _next;
        private ParameterSet[] _results;
        private int _maximum;

        public GridSearchSweeper(MLContext context, IValueGenerator[] valueGenerators, int maximum = 10000)
        {
            var option = new RandomGridSweeper.Options();
            this._maximum = maximum;
            this._gridSweeper = new RandomGridSweeper(context, option, valueGenerators);
            this._results = this._gridSweeper.ProposeSweeps(maximum);
        }

        public ParameterSet Current => this._next;

        object IEnumerator.Current => this._next;

        public void Dispose()
        {
            return;
        }

        public IEnumerator<ParameterSet> GetEnumerator()
        {
            return this;
        }

        public bool MoveNext()
        {
            if (this._maximum <= 0)
            {
                return false;
            }

            this._next = this._results[this._maximum - 1];
            this._maximum -= 1;
            return true;
        }

        public void Reset()
        {
            return;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this;
        }

        public void AddRunHistory(IRunResult input)
        {
            return;
        }
    }
}
