#ifndef _READTESTDATA_H_
#define _READTESTDATA_H_

#include <kaleidoscope.h>
#include <stdio.h>

int readTransformInfo(const char *path, TransformationInfo *dataPtr)
{
	FILE *fptr = NULL;
	int ctr = 0, val1, val2, val3, val4;

	if (!path || !dataPtr)
		return -1;
	if ((fptr = fopen(path, "rb")) == NULL)
		return -2;

	while (fscanf(fptr, "%d %d %d %d\n", &val1, &val2, &val3, &val4) != EOF)
	{
		dataPtr[ctr].srcLocation.x = val1;
		dataPtr[ctr].srcLocation.y = val2;
		dataPtr[ctr].dstLocation.x = val3;
		dataPtr[ctr].dstLocation.y = val4;
	}

	return ctr;
}

#endif // _READTESTDATA_H_
