import { TestBed } from '@angular/core/testing';

import { DigitApiService } from './digit-api.service';

describe('DigitApiService', () => {
  let service: DigitApiService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(DigitApiService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
