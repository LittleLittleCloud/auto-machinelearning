// <copyright file="Classification.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Experiment;
using MLNet.AutoPipeline.Metric;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLNet.Expert.AutoML
{
    public class Classification
    {
        private MLContext context;
        private Option option;
        private ClassificationExpert classificationExpert;
        private Dictionary<AutoMLTrainingState, ExperimentResult> history;
        private double timeLeftInSecond;

        public Classification(MLContext context, Option option)
        {
            this.context = context;
            this.history = new Dictionary<AutoMLTrainingState, ExperimentResult>();
            this.option = option;
            this.timeLeftInSecond = option.MaximumTrainingTime;
            this.classificationExpert = new ClassificationExpert(context, this.option.ClassificationExpertOption);
        }

        public async Task<ExperimentResult> TrainAsync(IDataView train, IDataView validate, IProgress<IterationInfo> reporter = null, CancellationToken ct = default)
        {
            Contract.Requires(train.Schema.Equals(validate.Schema));
            var stopWatch = new Stopwatch();

            var columnPicker = new ColumnPicker(train, new ColumnPicker.Option()
            {
                CatagoricalColumns = this.option.CatagoryColumns,
                IgnoreColumns = this.option.IgnoreColumns,
                LabelColumn = this.option.LabelColumn,
            });

            // start
            var trainers = this.classificationExpert.Propose(this.option.LabelColumn, PipelineBuilder.FEATURECOLUMNNAME);
            var initState = new AutoMLTrainingState(trainers as EstimatorNodeGroup);
            initState.Transformers.Add(columnPicker.LabelColumn, Util.CreateUnSweepableNode(this.context.Transforms.Conversion.MapValueToKey(this.option.LabelColumn, this.option.LabelColumn), estimatorName: "MapValueToKey") as INode);
            var experimentOption = new Experiment.Option()
            {
                ScoreMetric = this.option.ScoreMetric,
                Label = this.option.LabelColumn,
                Iteration = 3,
            };

            foreach (var column in columnPicker.SelectColumn(initState.Columns, this.option.BeamSearch))
            {
                (var state, var result) = await this.CreateExperimentBasedOnStateAndTrainAsync(
                                            train,
                                            validate,
                                            initState,
                                            column,
                                            columnPicker,
                                            experimentOption,
                                            trainers as EstimatorNodeGroup,
                                            reporter,
                                            ct);
                this.history.Add(state, result);
                this.timeLeftInSecond -= result.TrainingTime;
                if (this.timeLeftInSecond < 0)
                {
                    break;
                }
            }

            do
            {
                if (this.timeLeftInSecond < 0)
                {
                    break;
                }

                IEnumerable<(AutoMLTrainingState, DataViewSchema.Column?)> candidates;
                if (this.option.ScoreMetric.IsMaximizing)
                {
                    candidates = this.history.OrderByDescending(x => x.Value.BestIteration.ScoreMetric.Score)
                                             .Select(x => (x.Key, (DataViewSchema.Column?)columnPicker.SelectColumn(x.Key.Columns, 1).FirstOrDefault()))
                                             .Where(x => x.Item2 != null)
                                             .Take(this.option.BeamSearch);
                }
                else
                {
                    candidates = this.history.OrderByDescending(x => x.Value.BestIteration.ScoreMetric.Score)
                                             .Select(x => (x.Key, (DataViewSchema.Column?)columnPicker.SelectColumn(x.Key.Columns, 1).FirstOrDefault()))
                                             .Where(x => x.Item2 != null)
                                             .Take(this.option.BeamSearch);
                }

                if (candidates.Count() == 0)
                {
                    break;
                }

                foreach (var candidate in candidates)
                {
                    (var state, var result) = await this.CreateExperimentBasedOnStateAndTrainAsync(
                            train,
                            validate,
                            candidate.Item1,
                            candidate.Item2.Value,
                            columnPicker,
                            experimentOption,
                            trainers as EstimatorNodeGroup,
                            reporter,
                            ct);
                    this.history.Add(state, result);
                    this.timeLeftInSecond -= result.TrainingTime;
                    if (this.timeLeftInSecond < 0)
                    {
                        break;
                    }
                }
            } while (true);

            return this.HandleExperimentResultAndReturn(this.history);
        }

        private ExperimentResult HandleExperimentResultAndReturn(Dictionary<AutoMLTrainingState, ExperimentResult> history)
        {
            if (this.option.ScoreMetric.IsMaximizing)
            {
                return history.OrderByDescending(x => x.Value.BestIteration.ScoreMetric.Score)
                                   .Take(1)
                                   .Select(x => x.Value)
                                   .First();
            }
            else
            {
                return history.OrderByDescending(x => x.Value.BestIteration.ScoreMetric.Score)
                   .Take(1)
                   .Select(x => x.Value)
                   .First();
            }
        }

        private async Task<(AutoMLTrainingState, ExperimentResult)> CreateExperimentBasedOnStateAndTrainAsync(
            IDataView train,
            IDataView validate,
            AutoMLTrainingState currentState,
            DataViewSchema.Column column,
            ColumnPicker columnPicker,
            Experiment.Option experimentOption,
            EstimatorNodeGroup trainers,
            IProgress<IterationInfo> reporter = null,
            CancellationToken ct = default)
        {
            var expert = columnPicker.GetColumnType(column) switch
            {
                ColumnType.Numeric => NumericFeatureExpert.GetDefaultNumericFeatureExpert(this.context),
                _ => throw new Exception("Expert not found"),
            };
            var pipeline = PipelineBuilder.BuildPipelineFromStateWithNewColumn(this.context, currentState, column, expert, column.Name);
            var experiment = new Experiment(this.context, pipeline, experimentOption);
            var experimentResult = await experiment.TrainAsync(train, validate, reporter, ct);
            var transformers = experimentResult.BestIteration.SweepablePipeline.Nodes;

            var selectedTransformer = transformers[transformers.Count - 3];
            var transforms = currentState.Transformers.Select(x => x).ToDictionary(x => x.Key, x => x.Value);
            transforms.Add(column, selectedTransformer);
            var inputOutputPair = currentState.InputOutputColumnPairs.Select(x => x).ToList();
            inputOutputPair.Add(new InputOutputColumnPair(column.Name, column.Name));
            var state = new AutoMLTrainingState(transforms, inputOutputPair, trainers as EstimatorNodeGroup);
            return (state, experimentResult);
        }

        public class Option
        {
            public double MaximumTrainingTime { get; set; }

            public string[] CatagoryColumns { get; set; } = new string[0];

            public string[] IgnoreColumns { get; set; } = new string[0];

            public string LabelColumn { get; set; }

            public IMetric ScoreMetric { get; set; }

            public int BeamSearch { get; set; } = 3;

            public ClassificationExpert.Option ClassificationExpertOption { get; set; } = new ClassificationExpert.Option();
        }
    }
}
