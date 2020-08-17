// <copyright file="AutoMLTrainingState.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;

namespace MLNet.Expert.AutoML
{
    public class AutoMLTrainingState
    {
        public AutoMLTrainingState(IEnumerable<INode> trainers)
        {
            this.Trainers = trainers;
            this.InputOutputColumnPairs = new List<InputOutputColumnPair>();
            this.Transformers = new Dictionary<DataViewSchema.Column, INode>();
        }

        public AutoMLTrainingState(Dictionary<DataViewSchema.Column, INode> transformers, List<InputOutputColumnPair> inputOutputColumnPairs, IEnumerable<INode> trainers)
        {
            this.Trainers = trainers;
            this.Transformers = transformers;
            this.InputOutputColumnPairs = inputOutputColumnPairs;
        }

        public Dictionary<DataViewSchema.Column, INode> Transformers { get; private set; }

        public List<DataViewSchema.Column> Columns
        {
            get => this.Transformers.Keys.ToList();
        }

        public List<InputOutputColumnPair> InputOutputColumnPairs { get; private set; }

        public IEnumerable<INode> Trainers { get; private set; }
    }
}
