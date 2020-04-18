// <copyright file="RandomSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.Sweeper;
using System.Collections;
using System.Collections.Generic;

namespace MLNet.AutoPipeline
{
    internal class RandomSweeper : ISweeper
    {
        private ParameterSet _next;
        private int _maximum;

        private UniformRandomSweeper _uniformSweeper;

        public RandomSweeper(MLContext mlContext, IValueGenerator[] valueGenerators, int maximum = 100)
        {
            this._maximum = maximum;
            this._uniformSweeper = new UniformRandomSweeper(mlContext, new SweeperOptionBase(), valueGenerators);
        }

        public ParameterSet Current => this._next;

        object IEnumerator.Current => this.Current;

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

            this._maximum -= 1;
            var nextArray = this._uniformSweeper.ProposeSweeps(1);
            this._next = nextArray[0];
            return true;
        }

        public void Reset()
        {
            return;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public void AddRunHistory(IRunResult input)
        {
            return;
        }
    }
}
