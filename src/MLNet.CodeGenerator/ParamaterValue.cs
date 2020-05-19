using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace MLNet.CodeGenerator
{
    public enum ParamaterValueType
    {
        /// <summary>
        /// String type.
        /// </summary>
        String = 1,

        /// <summary>
        /// Float type.
        /// </summary>
        Float = 2,

        /// <summary>
        /// Int32 type.
        /// </summary>
        Int32 = 3,

        /// <summary>
        /// Enum type.
        /// </summary>
        Enum = 4,

        /// <summary>
        /// String array type.
        /// </summary>
        StringArray = 5,

        /// <summary>
        /// Float array type.
        /// </summary>
        FloatArray = 6,

        /// <summary>
        /// Int32 array type.
        /// </summary>
        Int32Array = 7,
    }

    public class ParamaterValue : ICodeGenNode
    {
        public ParamaterValueType Type { get; private set; }

        public string Value { get; private set; }

        private ParamaterValue() { }

        public static ParamaterValue Create<T>(T value)
        {
            var pv = new ParamaterValue();

            if (value is string)
            {
                pv.Value = $"\"{value as string}\"";
                pv.Type = ParamaterValueType.String;
                return pv;
            }

            if (value is int)
            {
                pv.Value = (value as int?).ToString();
                pv.Type = ParamaterValueType.Int32;
                return pv;
            }

            if (value is float || value is double)
            {
                pv.Value = $"{Convert.ToString(value)}F";
                pv.Type = ParamaterValueType.Float;
                return pv;
            }

            if (typeof(T).IsEnum)
            {
                pv.Value = $"{typeof(T).FullName}.{value}";
                pv.Type = ParamaterValueType.Enum;
                return pv;
            }

            if (value is IEnumerable<string>)
            {
                var strs = string.Join(",", (value as IEnumerable<string>).Select(x => $"\"{x}\""));
                pv.Value = $"new string[]{{{strs}}}";
                pv.Type = ParamaterValueType.StringArray;
                return pv;
            }

            if (value is IEnumerable<int>)
            {
                var strs = string.Join(",", (value as IEnumerable<int>).Select(x => x.ToString()));
                pv.Value = $"new int[]{{{strs}}}";
                pv.Type = ParamaterValueType.Int32Array;
                return pv;
            }

            if (value is IEnumerable<float> || value is IEnumerable<double>)
            {
                var strList = new List<string>();

                foreach (var val in value as IEnumerable)
                {
                    strList.Add($"{Convert.ToString(val)}F");
                }

                var strs = string.Join(",", strList);
                pv.Value = $"new float[]{{{strs}}}";
                pv.Type = ParamaterValueType.FloatArray;
                return pv;
            }

            throw new Exception("Unrecognize ParamaterValue type");
        }

        public string GeneratorCode()
        {
            return this.Value;
        }
    }
}
