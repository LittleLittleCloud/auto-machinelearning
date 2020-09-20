// <copyright file="ITrainingService.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLNet.AutoPipeline
{
    internal interface ITrainingService
    {
        Task<IterationInfo> StartTrainingAsync(IDataView train, IDataView validate, CancellationToken ct = default, IProgress<IterationInfo> reporter = null);

        Task<IterationInfo> StartTrainingCVAsync(IDataView train, int fold = 5, CancellationToken ct = default, IProgress<IterationInfo> reporter = null);
    }
}
