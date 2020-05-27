using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.AutoPipeline.Experiment
{
    /// <summary>
    /// Experiment Result for <see cref="Experiment"/>.
    /// </summary>
    public class ExperimentResult
    {
        private IList<IterationInfo> runHistories;
        private IterationInfo bestTrainingRound;
        private static object _lock = new Object();

        public ITransformer BestModel { get; private set; }

        public IterationInfo BestIteration { get => this.bestTrainingRound; }

        public double TrainingTime { get => this.runHistories.Select(x => x.TrainingTime).Sum(); }

        public ExperimentResult()
        {
            this.runHistories = new List<IterationInfo>();
        }

        /// <summary>
        /// Get all train rounds after training.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<IterationInfo> GetRunHistories()
        {
            return this.runHistories;
        }

        internal void AddRunHistory(IterationInfo info, ITransformer model)
        {
            lock (ExperimentResult._lock)
            {
                this.runHistories.Add(info);

                if (this.bestTrainingRound is null || info > this.bestTrainingRound)
                {
                    this.bestTrainingRound = info;
                    this.BestModel = model;
                }
            }
        }
    }
}
