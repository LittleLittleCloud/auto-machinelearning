using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Experiment;
using MLNet.Expert.AutoML;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;

namespace MLNet.Expert
{
    public class PipelineBuilder
    {
        public static string FEATURECOLUMNNAME = "features";

        public static EstimatorNodeChain BuildPipelineFromState(MLContext context, AutoMLTrainingState state)
        {
            var pipeline = new EstimatorNodeChain();
            pipeline.Append(state.Transformers.Select(x => new EstimatorSingleNode(x.Value)));

            var concatFeaturesTransformer = PipelineBuilder.BuildConcatFeaturesTransformer(context, state.Columns, state.InputOutputColumnPairs, PipelineBuilder.FEATURECOLUMNNAME);
            pipeline.Append(new EstimatorSingleNode(concatFeaturesTransformer));

            pipeline.Append(state.Trainers);
            return pipeline;
        }

        public static EstimatorNodeChain BuildPipelineFromColumns(MLContext context, IEnumerable<DataViewSchema.Column> columns, ColumnPicker columnPicker, IEnumerable<InputOutputColumnPair> inputOutputColumnPairs, EstimatorNodeGroup trainers)
        {
            Contract.Requires(columns!.Count() == inputOutputColumnPairs!.Count());
            var pipeline = new EstimatorNodeChain();
            foreach (var column in columns)
            {
                var inputOutputColumnPair = inputOutputColumnPairs.Where(x => x.InputColumnName == column.Name).First();
                var expert = columnPicker.GetColumnType(column) switch
                {
                    ColumnType.Numeric => NumericFeatureExpert.GetDefaultNumericFeatureExpert(context),
                    _ => throw new Exception("Expert not found"),
                };

                pipeline.Append(expert.Propose(column.Name, inputOutputColumnPair.OutputColumnName));
            }

            var concatFeaturesTransformer = PipelineBuilder.BuildConcatFeaturesTransformer(context, columns, inputOutputColumnPairs, PipelineBuilder.FEATURECOLUMNNAME);
            pipeline.Append(new EstimatorSingleNode(concatFeaturesTransformer));
            pipeline.Append(trainers);

            return pipeline;
        }

        public static EstimatorNodeChain BuildPipelineFromStateWithNewColumn(MLContext context, AutoMLTrainingState state, DataViewSchema.Column column, ITransformExpert expert, string outputColumnName)
        {
            var pipeline = new EstimatorNodeChain();

            pipeline.Append(state.Transformers.Select(x => new EstimatorSingleNode(x.Value)));
            pipeline.Append(expert.Propose(column.Name, outputColumnName));

            var inputOutputPair = state.InputOutputColumnPairs.ToList();
            inputOutputPair.Add(new InputOutputColumnPair(outputColumnName, column.Name));

            var columns = state.Columns.ToList();
            columns.Add(column);

            var concatFeaturesTransformer = PipelineBuilder.BuildConcatFeaturesTransformer(context, columns.ToArray(), inputOutputPair.ToArray(), PipelineBuilder.FEATURECOLUMNNAME);
            pipeline.Append(new EstimatorSingleNode(concatFeaturesTransformer));

            pipeline.Append(state.Trainers);
            return pipeline;
        }

        private static ISweepablePipelineNode BuildConcatFeaturesTransformer(MLContext context, IEnumerable<DataViewSchema.Column> columns, IEnumerable<InputOutputColumnPair> inputOutputColumnPairs, string featureColumnName = "Features")
        {
            Contract.Requires(columns != null && inputOutputColumnPairs != null);
            var inputColumnNames = inputOutputColumnPairs.Select(x => x.InputColumnName);
            var columnNames = columns.Select(c => c.Name);
            foreach (var name in inputColumnNames)
            {
                Contract.Requires(columnNames.Contains(name), $"input column {name} not exist");
            }

            var outputColumnNames = inputOutputColumnPairs.Select(x => x.OutputColumnName);
            var concatFeaturesTransformer = context.Transforms.Concatenate(featureColumnName, outputColumnNames.ToArray());
            return Util.CreateUnSweepableNode(concatFeaturesTransformer);
        }
    }
}
