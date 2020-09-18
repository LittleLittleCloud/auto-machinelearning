// <copyright file="SingleEstimatorSweepablePipelineDataContract.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;

namespace MLNet.Expert.Contract
{
    [DataContract]
    public class SingleEstimatorSweepablePipelineDataContract
    {
        [DataMember]
        public List<SweepableEstimatorDataContract> Estimators { get; set; }
    }
}
