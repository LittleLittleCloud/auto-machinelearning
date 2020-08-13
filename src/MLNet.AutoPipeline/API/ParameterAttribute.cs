// <copyright file="ParameterAttribute.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Specifies the parameter used to build options.
    /// </summary>
    [System.AttributeUsage(AttributeTargets.Field, AllowMultiple =false, Inherited =false)]
    public class ParameterAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ParameterAttribute"/> class.
        /// </summary>
        /// <param name="name">parameter name.</param>
        public ParameterAttribute(string name)
        {
            this.Name = name;
        }

        public ParameterAttribute() { }

        public string Name { get; private set; }
    }
}
