// <copyright file="SweepableEstimatorDataContract.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;

namespace MLNet.AutoPipeline
{
    [DataContract]
    public class SweepableEstimatorDataContract
    {
        [DataMember]
        public string EstimatorName { get; protected set; }

        // TODO
        // use dictionary instead of string array.
        [DataMember]
        public string[] InputColumns { get; protected set; }

        // TODO
        // use dictionary instead of string array.
        [DataMember]
        public string[] OutputColumns { get; protected set; }

        [DataMember]
        public TransformerScope Scope { get; protected set; }
    }
}
