using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
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

        public static SweepablePipeline BuildPipelineFromState(MLContext context, AutoMLTrainingState state)
        {
            var pipeline = new SweepablePipeline();
            pipeline.Append(state.Transformers.Values.ToArray());

            var concatFeaturesTransformer = PipelineBuilder.BuildConcatFeaturesTransformer(context, state.Columns, state.InputOutputColumnPairs, PipelineBuilder.FEATURECOLUMNNAME);
            pipeline.Append(concatFeaturesTransformer);

            pipeline.Append(state.Trainers.ToArray());
            return pipeline;
        }

        public static SweepablePipeline BuildPipelineFromColumns(MLContext context, IEnumerable<DataViewSchema.Column> columns, ColumnPicker columnPicker, IEnumerable<InputOutputColumnPair> inputOutputColumnPairs, SweepableEstimatorBase[] trainers)
        {
            Contract.Requires(columns!.Count() == inputOutputColumnPairs!.Count());
            var pipeline = new SweepablePipeline();
            foreach (var column in columns)
            {
                var inputOutputColumnPair = inputOutputColumnPairs.Where(x => x.InputColumnName == column.Name).First();
                var expert = columnPicker.GetColumnType(column) switch
                {
                    ColumnType.Numeric => NumericFeatureExpert.GetDefaultNumericFeatureExpert(context),
                    _ => throw new Exception("Expert not found"),
                };

                pipeline.Append(expert.Propose(column.Name, inputOutputColumnPair.OutputColumnName).ToArray());
            }

            var concatFeaturesTransformer = PipelineBuilder.BuildConcatFeaturesTransformer(context, columns, inputOutputColumnPairs, PipelineBuilder.FEATURECOLUMNNAME);
            pipeline.Append(concatFeaturesTransformer);
            pipeline.Append(trainers);

            return pipeline;
        }

        public static SweepablePipeline BuildPipelineFromStateWithNewColumn(MLContext context, AutoMLTrainingState state, DataViewSchema.Column column, ITransformExpert expert, string outputColumnName)
        {
            var pipeline = new SweepablePipeline();

            pipeline.Append(state.Transformers.Values.ToArray());
            pipeline.Append(expert.Propose(column.Name, outputColumnName).ToArray());

            var inputOutputPair = state.InputOutputColumnPairs.ToList();
            inputOutputPair.Add(new InputOutputColumnPair(outputColumnName, column.Name));

            var columns = state.Columns.ToList();
            columns.Add(column);

            var concatFeaturesTransformer = PipelineBuilder.BuildConcatFeaturesTransformer(context, columns.ToArray(), inputOutputPair.ToArray(), PipelineBuilder.FEATURECOLUMNNAME);
            pipeline.Append(concatFeaturesTransformer);

            pipeline.Append(state.Trainers.ToArray());
            return pipeline;
        }

        private static SweepableEstimatorBase BuildConcatFeaturesTransformer(MLContext context, IEnumerable<DataViewSchema.Column> columns, IEnumerable<InputOutputColumnPair> inputOutputColumnPairs, string featureColumnName = "Features")
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
