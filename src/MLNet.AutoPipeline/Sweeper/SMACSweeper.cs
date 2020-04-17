using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Sweeper;

namespace Microsoft.ML.AutoPipeline
{
    internal class SMACSweeper : ISweeper
    {
        private SmacSweeper _smacSweeper;
        private int _maximum;
        private ParameterSet _next;
        private List<IRunResult> _runHistory;

        public ParameterSet Current => _next;

        object IEnumerator.Current => Current;

        public SMACSweeper(MLContext context, IValueGenerator[] valueGenerators, int maximum = 100)
        {
            _maximum = maximum;
            _runHistory = new List<IRunResult>();
            var option = new SmacSweeper.Options();
            _smacSweeper = new SmacSweeper(context, option, valueGenerators);
        }

        public void AddRunHistory(IRunResult input)
        {
            this._runHistory.Add(input);
        }

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
            if (_maximum <= 0)
            {
                return false;
            }

            _maximum -= 1;
            if (_runHistory.Count > 20)
            {
                _next = this._smacSweeper.ProposeSweeps(1, _runHistory)[0];
            }
            else
            {
                _next = this._smacSweeper.ProposeSweeps(1)[0];
            }
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
    }
}
