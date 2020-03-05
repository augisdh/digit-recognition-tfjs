import { Component, OnInit } from "@angular/core";
import { DigitApiService } from "src/app/service/digit-api.service";

@Component({
  selector: "app-home",
  templateUrl: "./home.component.html",
  styleUrls: ["./home.component.css"]
})
export class HomeComponent implements OnInit {
  constructor(private digitApi: DigitApiService) {}

  ngOnInit(): void {}

  runModel() {
    this.digitApi.run();
  }
}
