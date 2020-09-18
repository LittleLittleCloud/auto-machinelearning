// <copyright file="SweepableEstimatorDataContract.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;

namespace MLNet.Expert.Contract
{
    [DataContract]
    public class SweepableEstimatorDataContract
    {
        [DataMember]
        public string EstimatorName { get; set; }

        // TODO
        // use dictionary instead of string array.
        [DataMember]
        public string[] InputColumns { get; set; }

        // TODO
        // use dictionary instead of string array.
        [DataMember]
        public string[] OutputColumns { get; set; }

        [DataMember]
        public TransformerScope Scope { get; set; }
    }
}
