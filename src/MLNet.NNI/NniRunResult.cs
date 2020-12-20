// <copyright file="NniRunResult.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.NNI
{
    public class NniRunResult
    {
        public Dictionary<string, string> Parameters { get; set; }

        public string PipelineContract { get; set; }

        public Dictionary<string, float> Metrics { get; set; }

        public string ModelPath { get; set; }

        public float Duration { get; set; }
    }
}
