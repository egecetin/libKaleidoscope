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

	while (fscanf(fptr, "%d,%d,%d,%d", &val1, &val2, &val3, &val4) == 4)
	{
		dataPtr[ctr].srcLocation.x = val1;
		dataPtr[ctr].srcLocation.y = val2;
		dataPtr[ctr].dstLocation.x = val3;
		dataPtr[ctr].dstLocation.y = val4;
		++ctr;
	}

	fclose(fptr);
	return ctr;
}

int writeTransformInfo(const char *path, TransformationInfo *dataPtr, unsigned long long len)
{
	int idx = 0;
	FILE *fptr = NULL;

	if (!path || !dataPtr)
		return -1;
	if ((fptr = fopen(path, "wb")) == NULL)
		return -2;

	for (idx = 0; idx < len; ++idx)
	{
		fprintf(fptr, "%d,%d,%d,%d\n", dataPtr[idx].srcLocation.x, dataPtr[idx].srcLocation.y,
				dataPtr[idx].dstLocation.x, dataPtr[idx].dstLocation.y);
	}

	fclose(fptr);
	return 0;
}

#endif // _READTESTDATA_H_
